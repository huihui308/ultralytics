#
#
# cmd:
#       rm -rf result/*;python3 v8_inference.py --input=/home/david/code/yolo/ONNX-YOLOv8-Object-Detection/doc/input --output_dir=./result --interval=250
#
import cv2
from ultralytics import YOLO
import os, sys, math, shutil, random, datetime, signal, argparse


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
        "--interval",
        type = int,
        required = False,
        default = 25,
        help = "Parse every interval frame when input is a video file."
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
    return parser.parse_args(args)


def inference_img(
        input_file, 
        output_dir, 
        conf_thres, 
        iou_thres,
        yolov8_detector
)->None:
    file_path, file_type = os.path.splitext(input_file)
    save_file = os.path.join(output_dir, file_path.split('/')[-1] + file_type)
    #print(save_file)
    img = cv2.imread(input_file, 1)
    # Detect Objects
    results = yolov8_detector(frame, conf=conf_thres, iou=iou_thres)
    # Visualize the results on the frame
    annotated_frame = results[0].plot(line_width =3, font_size=6.0, masks=True)
    cv2.imwrite(save_file, annotated_frame)
    return


def inference_video_file(
        input_file, 
        output_dir, 
        interval, 
        conf_thres, 
        iou_thres,
        yolov8_detector
)->None:
    file_path, file_type = os.path.splitext(input_file)
    prGreen('Video file is {}'.format(input_file))
    videoCapture = cv2.VideoCapture(input_file)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    prYellow('Video info: {} {} {}'.format(fps, size, fNUMS))
    # read frames
    success = True
    #success, frame = videoCapture.read()
    loop_cnt = 0
    infer_cnt = 0
    success = True
    # Loop through the video frames
    while videoCapture.isOpened() and success:
        # Read a frame from the video
        success, frame = videoCapture.read()
        loop_cnt += 1
        if (interval != 0) and ((loop_cnt % interval) != 0):
            continue
        if success:
            # Run YOLOv8 inference on the frame
            results = yolov8_detector(frame, conf=conf_thres, iou=iou_thres)
            # Visualize the results on the frame
            annotated_frame = results[0].plot(line_width =3, font_size=6.0, masks=True)
            # Display the annotated frame
            #cv2.imshow("YOLOv8 Inference", annotated_frame)
            # Break the loop if 'q' is pressed
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break
            save_file = os.path.join(output_dir, file_path.split('/')[-1] + '_' + str(infer_cnt).zfill(20) + '.jpg')
            infer_cnt += 1
            cv2.imwrite(save_file, annotated_frame)
            prGreen('Save inference result: {}'.format(save_file))
        else:
            # Break the loop if the end of the video is reached
            break
    # Release the video capture object and close the display window
    videoCapture.release()
    prGreen('Read images count: {}, inference images count:{}'.format(loop_cnt, infer_cnt))
    return


def deal_input_file(
        input_file, 
        output_dir, 
        interval, 
        conf_thres, 
        iou_thres,
        yolov8_detector
)->None:
    file_path, file_type = os.path.splitext(input_file)
    if file_path.split('/')[-1] == '.gitignore':
        prGreen('It is a \'.gitignore\' file, return')
        return
    if file_type in ('.jpg', '.png'):
        inference_img(input_file, output_dir, conf_thres, iou_thres, yolov8_detector)
    elif file_type in ('.mp4'):
        inference_video_file(input_file, output_dir, interval, conf_thres, iou_thres, yolov8_detector)
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
                deal_input_file(deal_file, args.output_dir, args.interval, args.conf_thres, args.iou_thres, yolov8_detector)
    elif os.path.isfile(args.input):
        #print("it's a normal file")
        deal_input_file(args.input, args.output_dir, args.interval, args.conf_thres, args.iou_thres, yolov8_detector)
    else:
        prRed("it's a special file(socket,FIFO,device file)")
        return
    return


if __name__ == "__main__":
    main_func()