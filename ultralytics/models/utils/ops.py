# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


class HungarianMatcher(nn.Module):
    """
    A module implementing the HungarianMatcher, which is a differentiable module to solve the assignment problem in an
    end-to-end fashion.

    HungarianMatcher performs optimal assignment over the predicted and ground truth bounding boxes using a cost
    function that considers classification scores, bounding box coordinates, and optionally, mask predictions.

    Attributes:
        cost_gain (dict): Dictionary of cost coefficients: 'class', 'bbox', 'giou', 'mask', and 'dice'.
        use_fl (bool): Indicates whether to use Focal Loss for the classification cost calculation.
        with_mask (bool): Indicates whether the model makes mask predictions.
        num_sample_points (int): The number of sample points used in mask cost calculation.
        alpha (float): The alpha factor in Focal Loss calculation.
        gamma (float): The gamma factor in Focal Loss calculation.

    Methods:
        forward(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None): Computes the
            assignment between predictions and ground truths for a batch.
        _cost_mask(bs, num_gts, masks=None, gt_mask=None): Computes the mask cost and dice cost if masks are predicted.
    """

    def __init__(self, cost_gain=None, use_fl=True, with_mask=False, num_sample_points=12544, alpha=0.25, gamma=2.0):
        """Initializes HungarianMatcher with cost coefficients, Focal Loss, mask prediction, sample points, and alpha
        gamma factors.
        """
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
        """
        Forward pass for HungarianMatcher. This function computes costs based on prediction and ground truth
        (classification cost, L1 cost between boxes and GIoU cost between boxes) and finds the optimal matching between
        predictions and ground truth based on these costs.

        Args:
            pred_bboxes (Tensor): Predicted bounding boxes with shape [batch_size, num_queries, 4].
            pred_scores (Tensor): Predicted scores with shape [batch_size, num_queries, num_classes].
            gt_cls (torch.Tensor): Ground truth classes with shape [num_gts, ].
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape [num_gts, 4].
            gt_groups (List[int]): List of length equal to batch size, containing the number of ground truths for
                each image.
            masks (Tensor, optional): Predicted masks with shape [batch_size, num_queries, height, width].
                Defaults to None.
            gt_mask (List[Tensor], optional): List of ground truth masks, each with shape [num_masks, Height, Width].
                Defaults to None.

        Returns:
            (List[Tuple[Tensor, Tensor]]): A list of size batch_size, each element is a tuple (index_i, index_j), where:
                - index_i is the tensor of indices of the selected predictions (in order)
                - index_j is the tensor of indices of the corresponding selected ground truth targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        # [batch_size * num_queries, 4]
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        # Compute the classification cost
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores**self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # Compute the L1 cost between boxes
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # (bs*num_queries, num_gt)

        # Compute the GIoU cost between boxes, (bs*num_queries, num_gt)
        cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)

        # Final cost matrix
        C = (
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox
            + self.cost_gain["giou"] * cost_giou
        )
        # Compute the mask cost and dice cost
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        # Set invalid values (NaNs and infinities) to 0 (fixes ValueError: matrix contains invalid numeric entries)
        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)  # (idx for queries, idx for gt)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups[k])
            for k, (i, j) in enumerate(indices)
        ]

    # This function is for future RT-DETR Segment models
    # def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
    #     assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
    #     # all masks share the same set of points for efficient matching
    #     sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
    #     sample_points = 2.0 * sample_points - 1.0
    #
    #     out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
    #     out_mask = out_mask.flatten(0, 1)
    #
    #     tgt_mask = torch.cat(gt_mask).unsqueeze(1)
    #     sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
    #     tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])
    #
    #     with torch.cuda.amp.autocast(False):
    #         # binary cross entropy cost
    #         pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
    #         neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
    #         cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
    #         cost_mask /= self.num_sample_points
    #
    #         # dice cost
    #         out_mask = F.sigmoid(out_mask)
    #         numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
    #         denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
    #         cost_dice = 1 - (numerator + 1) / (denominator + 1)
    #
    #         C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
    #     return C


"""
    加入噪声后，还需要注意的一点便是信息之间的是否可见问题，噪声 queries 是会和匈牙利匹配任务的 queries 拼接起来一起送入 transformer中的。在 transformer 中，它们会经过 attention 交互，这势必会得知一些信息，这是作弊行为，是绝对不允许的
    一、首先，如上所述，匈牙利匹配任务的 queries 肯定不能看到 DN 任务的 queries。
    二、其次，不同 dn group 的 queries 也不能相互看到。因为综合所有组来看，gt -> query 是 one-to-many 的，每个 gt 在
    每组都会有 1 个 query 拥有自己的信息。于是，对于每个 query 来说，在其它各组中都势必存在 1 个 query 拥有自己负责预测的那个 gt 的信息。
    三、接着，同一个 dn group 的 queries 是可以相互看的 。因为在每组内，gt -> query 是 one-to-one 的关系，对于每个 query 来说，其它 queries 都不会有自己 gt 的信息。
    四、最后，DN 任务的 queries 可以去看匈牙利匹配任务的 queries ，因为只有前者才拥有 gt 信息，而后者是“凭空构造”的（主要是先验，需要自己去学习）。
    总的来说，attention mask 的设计归纳为：
        1、匈牙利匹配任务的 queries 不能看到 DN任务的 queries；
        2、DN 任务中，不同组的 queries 不能相互看到；
        3、其它情况均可见
"""
def get_cdn_group(
    batch, num_classes, num_queries, class_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False
):
    """
    Get contrastive denoising training group. This function creates a contrastive denoising training group with positive
    and negative samples from the ground truths (gt). It applies noise to the class labels and bounding box coordinates,
    and returns the modified labels, bounding boxes, attention mask and meta information.
    Args:
        batch (dict): A dict that includes 'gt_cls' (torch.Tensor with shape [num_gts, ]), 'gt_bboxes'
            (torch.Tensor with shape [num_gts, 4]), 'gt_groups' (List(int)) which is a list of batch size length
            indicating the number of gts of each image.
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.Tensor): Embedding weights to map class labels to embedding space.
        num_dn (int, optional): Number of denoising. Defaults to 100.
        cls_noise_ratio (float, optional): Noise ratio for class labels. Defaults to 0.5.
        box_noise_scale (float, optional): Noise scale for bounding box coordinates. Defaults to 1.0.
        training (bool, optional): If it's in training mode. Defaults to False.
    Returns:
        (Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): The modified class embeddings,
            bounding boxes, attention mask and meta information for denoising. If not in training mode or 'num_dn'
            is less than or equal to 0, the function returns None for all elements in the tuple.
    """

    if (not training) or num_dn <= 0:
        return None, None, None, None
    gt_groups = batch["gt_groups"]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    # Pad gt to max_num of a batch
    bs = len(gt_groups)
    gt_cls = batch["cls"]  # (bs*num, )
    gt_bbox = batch["bboxes"]  # bs*num, 4
    b_idx = batch["batch_idx"]

    # Each group has positive and negative queries.   bs*num->total_num
    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, )
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )

    # Positive and negative mask
    # (bs*num*num_group, ), the second total_num*num_group part as negative samples
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        # Half of bbox prob
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        # Randomly put a new one here
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label  # 已经在GT中加入了噪声label

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)

        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 由GT bbox的w和h生成的四维张量 # 2*num_group*bs*num, 4

        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0  # 由GT bbox形状随机生成的[-1,1]的标识符
        rand_part = torch.rand_like(dn_bbox)  # 由GT bbox形状随机生成的随机数
        rand_part[neg_idx] += 1.0  # 对随机数索引后半部分的值加1
        rand_part *= rand_sign  # 随机数乘随机数标识符
        known_bbox += rand_part * diff  # 加入噪声的GT bbox
        known_bbox.clip_(min=0.0, max=1.0) # 将加入噪声的GT bbox限制在[0,1]
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # inverse sigmoid

    num_dn = int(max_nums * 2 * num_group)  # total denoising queries
    # class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])
    dn_cls_embed = class_embed[dn_cls]  # bs*num * 2 * num_group, 256
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)  # [2 num_dn 256]
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)  # [2 num_dn 4]

    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)  # 用于记录存在目标的索引

    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed  # BATCH中的目标数量不一样，用0补齐，以max_nums为一个周期
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # Match query cannot see the reconstruct
    attn_mask[num_dn:, :num_dn] = True
    # Reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
            attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i] = True
    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }

    return (
        padding_cls.to(class_embed.device),
        padding_bbox.to(class_embed.device),
        attn_mask.to(class_embed.device),
        dn_meta,
    )
