#
#
# cmd:
#       rm -rf result/*;python3 v8_inference.py --input=/home/david/code/yolo/ONNX-YOLOv8-Object-Detection/doc/input/NO1_highway.mp4 --model_file=./v8n_best.pt --output_dir=./result --save_video --save_image
#
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import os, sys, math, shutil, random, datetime, signal, argparse


TQDM_BAR_FORMAT = '{l_bar}{bar:40}| {n_fmt}/{total_fmt} {elapsed}'


def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))


def term_sig_handler(signum, frame)->None:
    sys.stdout.write('\r>> {}: \n\n\n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.write('\r>> {}: Catched singal: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), signum))
    sys.stdout.write('\r>> {}: \n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.flush()
    os._exit(0)
    return


def parse_args(args = None):
    """ parse the arguments. """
    parser = argparse.ArgumentParser(description = 'Prepare resized images/labels dataset for LPD')
    parser.add_argument(
        "--input",
        type = str,
        required = True,
        help = "Input directory or files which you want to inference."
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        required = True,
        help = "Ouput directory to save inference results."
    )
    parser.add_argument(
        "--model_file",
        type = str,
        required = False,
        default = './best.pt',
        help = "Model file path, such as: ./best.pt."
    )
    parser.add_argument(
        "--conf_thres",
        type = float,
        required = False,
        default = 0.4,
        help = "Yolov8 conf_thres."
    )
    parser.add_argument(
        "--iou_thres",
        type = float,
        required = False,
        default = 0.3,
        help = "Yolov8 iou_thres."
    )
    parser.add_argument(
        "--target_width",
        type = int,
        required = False,
        help = "Target width for resized images/labels."
    )
    parser.add_argument(
        "--target_height",
        type = int,
        required = False,
        help = "Target height for resized images/labels."
    )
    parser.add_argument(
        "--save_video",
        #type = bool,
        required = False,
        #default = True,
        action="store_true", 
        help = "If save result video when input is video file."
    )
    parser.add_argument(
        "--save_image",
        #type = bool,
        required = False,
        #default = True,
        action="store_true", 
        help = "If save result image when input is video file."
    )
    parser.add_argument(
        "--show_infer_log",
        #type = bool,
        required = False,
        #default = True,
        action="store_true", 
        help = "The switch of inference log."
    )
    parser.add_argument(
        "--interval",
        type = int,
        required = False,
        default = 25,
        help = "Parse every interval frame when input is a video file."
    )
    return parser.parse_args(args)


def inference_img(
        yolov8_detector, 
        input_file, 
        output_dir, 
        conf_thres, 
        iou_thres, 
        show_infer_log:bool
)->None:
    file_path, file_type = os.path.splitext(input_file)
    save_file = os.path.join(output_dir, file_path.split('/')[-1] + file_type)
    #print(save_file)
    img = cv2.imread(input_file, 1)
    # Detect Objects
    results = yolov8_detector(frame, conf=conf_thres, iou=iou_thres, verbose=show_infer_log)
    # Visualize the results on the frame
    annotated_frame = results[0].plot(line_width =3, font_size=6.0, masks=True)
    cv2.imwrite(save_file, annotated_frame)
    return


def inference_video_file(
        yolov8_detector, 
        input_file, 
        output_dir, 
        interval, 
        conf_thres, 
        iou_thres, 
        show_infer_log:bool, 
        save_video:bool, 
        save_image:bool
)->None:
    file_path, file_type = os.path.splitext(input_file)
    prGreen('Video file is {}'.format(input_file))
    video_capture = cv2.VideoCapture(input_file)
    fps_val = video_capture.get(cv2.CAP_PROP_FPS)
    video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_cnt = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    prYellow('Video info: {} {} {}'.format(fps_val, video_size, frame_cnt))
    # read frames
    loop_cnt = 0
    infer_cnt = 0
    # 视频编码格式
    #fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc('I','4','2','0')
    # 不能中途停止运行程序，否则保存的.mp4文件无法播放
    # (保存为.avi格式的视频，中途停止运行程序，保存的视频可以正常播放
    save_video_file = os.path.join(output_dir, file_path.split('/')[-1] + '_inference' + '.avi')
    video_out = cv2.VideoWriter(save_video_file, fourcc, fps_val, video_size)
    # Loop through the video frames
    for _ in tqdm( range( int(frame_cnt) ) ):
        success, frame = video_capture.read()   # Read a frame from the video
        if (not video_capture.isOpened()) or (success is False):
            break   # Break the loop if the end of the video is reached
        loop_cnt += 1
        if (interval != 0) and ((loop_cnt % interval) != 0):
            continue
        # Run YOLOv8 inference on the frame
        results = yolov8_detector(frame, conf=conf_thres, iou=iou_thres, verbose=show_infer_log)
        # Visualize the results on the frame
        annotated_frame = results[0].plot(line_width =3, font_size=6.0, masks=True)
        # Display the annotated frame
        #cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break
        if save_video is True:
            video_out.write(annotated_frame)
        if save_image is True:
            save_file = os.path.join(output_dir, file_path.split('/')[-1] + '_' + str(infer_cnt).zfill(20) + '.jpg')
            infer_cnt += 1
            cv2.imwrite(save_file, annotated_frame)
            prGreen('Save inference result: {}'.format(save_file))
    video_out.release()
    # Release the video capture object and close the display window
    video_capture.release()
    prGreen('Read images count: {}, inference images count:{}'.format(loop_cnt, infer_cnt))
    return


def deal_input_file(
        yolov8_detector, 
        input_file, 
        output_dir, 
        interval, 
        conf_thres, 
        iou_thres, 
        show_infer_log:bool, 
        save_video:bool, 
        save_image:bool
)->None:
    file_path, file_type = os.path.splitext(input_file)
    if file_path.split('/')[-1] == '.gitignore':
        prGreen('It is a \'.gitignore\' file, return')
        return
    if file_type in ('.jpg', '.png'):
        inference_img(yolov8_detector, input_file, output_dir, conf_thres, iou_thres, show_infer_log)
    elif file_type in ('.mp4'):
        inference_video_file(yolov8_detector, input_file, output_dir, interval, conf_thres, iou_thres, show_infer_log, save_video, save_image)
    else:
        prYellow('file_type({}) not support, return'.format(file_type))
        return
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    args.input = os.path.abspath(args.input)
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('input: {}, output_dir: {}, model_file: {}'.format(args.input, args.output_dir, args.model_file))
    # Load the YOLOv8 model
    yolov8_detector = YOLO(args.model_file)
    #------
    if os.path.isdir(args.input):
        #print("it's a directory")
        for root, dirs, files in os.walk(args.input):
            for lop_file in files:
                deal_file = os.path.join(root, lop_file)
                #print(deal_file)
                deal_input_file(yolov8_detector, deal_file, args.output_dir, args.interval, args.conf_thres, args.iou_thres, args.show_infer_log, args.save_video, args.save_image)
    elif os.path.isfile(args.input):
        #print("it's a normal file")
        deal_input_file(yolov8_detector, args.input, args.output_dir, args.interval, args.conf_thres, args.iou_thres, args.show_infer_log, args.save_video, args.save_image)
    else:
        prRed("it's a special file(socket,FIFO,device file)")
        return
    return


if __name__ == "__main__":
    main_func()