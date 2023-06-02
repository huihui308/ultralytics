#
#
#
# 
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
        default = './models/best.onnx',
        help = "Model file path, such as:models/best.onnx."
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




def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    #args = parse_args(args)
    # Load the YOLOv8 model
    model = YOLO('best.pt')

    # Open the video file
    video_path = "/home/david/code/yolo/ONNX-YOLOv8-Object-Detection/doc/input/NO1_highway.mp4"
    cap = cv2.VideoCapture(video_path)

    loop_cnt = 0
    infer_cnt = 0
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot(line_width =3, font_size=6.0, masks=True)

            # Display the annotated frame
            #cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break
            save_file = os.path.join('./result/' + str(infer_cnt).zfill(20) + '.jpg')
            infer_cnt += 1
            cv2.imwrite(save_file, annotated_frame)
            prGreen('Save inference result: {}'.format(save_file))
        else:
            # Break the loop if the end of the video is reached
            break
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main_func()