import numpy as np
import cv2
import sys
import argparse
import xml.dom.minidom
from pyannote.video import Video
import math


"""
This is an annotation tool for labelling objects within the frames of a video file.
"""

# Parse Command Line Arguments
parser = argparse.ArgumentParser(description='Input the video file name. ')
parser.add_argument('video_file', action="store", type=str)
args = parser.parse_args()

try:
    video_descriptor = args.video_file
    with open(video_descriptor, "r") as f:

        # Create a Video Capture obj and read from input file
        # Video Capture for cv2
        cap = cv2.VideoCapture(video_descriptor)
        # video capture for pyannote
        video = Video(video_descriptor)
        # video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

except IOError as e:
    print("Video not found", e)
    sys.exit(1)


def generate_new_doc(video_num, timestamp, frame_idx, width, height, depth):
    """
    Generate a new DOM object.
    <annotation>
        <size>
            <width>800</width>
            <height>410</height>
            <depth>3</depth>
        </size>
        <video>001</video>
        <timestamp>0.0</timestamp>
        <frameIdx>0</frameIdx>
    </annotation>

    :param video_num: video number
    :param timestamp: timestamp in the video file
    :param frame_idx: frame index
    :param width: width of the video frame
    :param height: height of the video frame
    :param depth: color depth of the video frame
    :return: a fresh DOM object
    """

    # Create a DOM obj
    new_doc = xml.dom.minidom.Document()
    # check if root exists
    # if len(new_doc.getElementsByTagName('annotation')) != 0:
    #     doc.removeChild(doc.getElementsByTagName('annotation')[0])

    # Create a root
    new_doc_root = new_doc.createElement('annotation')
    new_doc.appendChild(new_doc_root)

    # Add child to root node
    size_node = new_doc.createElement('size')
    new_doc_root.appendChild(size_node)

    width_node = new_doc.createElement('width')
    width_node.appendChild(new_doc.createTextNode(str(width)))
    size_node.appendChild(width_node)
    height_node = new_doc.createElement('height')
    height_node.appendChild(new_doc.createTextNode(str(height)))
    size_node.appendChild(height_node)
    depth_node = new_doc.createElement('depth')
    depth_node.appendChild(new_doc.createTextNode(str(depth)))
    size_node.appendChild(depth_node)

    video_node = new_doc.createElement('video')
    video_node.appendChild(new_doc.createTextNode('{:03d}'.format(int(video_num))))
    new_doc_root.appendChild(video_node)
    ts_node = new_doc.createElement('timestamp')
    ts_node.appendChild(new_doc.createTextNode(str(timestamp)))
    new_doc_root.appendChild(ts_node)
    idx_node = new_doc.createElement('frameIdx')
    idx_node.appendChild(new_doc.createTextNode(str(frame_idx)))
    new_doc_root.appendChild(idx_node)

    return new_doc


def prepareObj(obj):
    """
    Generate a new object

    <object>
        <name>car</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <box>
            <xmin>70</xmin>
            <ymin>232</ymin>
            <xmax>123</xmax>
            <ymax>269</ymax>
        </box>
     </object>

     or,

    <ignored>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <box>
            <xmin>399</xmin>
            <ymin>127</ymin>
            <xmax>539</xmax>
            <ymax>202</ymax>
        </box>
    </ignored>

    :param bbox: bounding box
    :param objType: 'car', or 'ignored'
    :return: DOM node
    """

    xmin, ymin, xmax, ymax = obj['box']
    # Prepare the car object 
    obj_dict = {
        'name': obj['type'],
        'pose': 'Unspecified',
        'truncated': '0',
        'difficult': '0',
        'box': {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }
    }
    
    # Create nodes
    object_node = doc.createElement('object' if obj['type'] == 'car' else 'ignored')

    if obj['type'] == 'car':
        name_node = doc.createElement('name')
        name_node.appendChild(doc.createTextNode(obj_dict['name']))
        object_node.appendChild(name_node)
        pose_node = doc.createElement('pose')
        pose_node.appendChild(doc.createTextNode(obj_dict['pose']))
        object_node.appendChild(pose_node)

    node_truncated = doc.createElement('truncated')
    node_truncated.appendChild(doc.createTextNode(obj_dict['truncated']))
    node_difficult = doc.createElement('difficult')
    node_difficult.appendChild(doc.createTextNode(obj_dict['difficult']))
                
    node_box = doc.createElement('box')
    for pos in ['xmin', 'ymin', 'xmax', 'ymax']:
        node_pos = doc.createElement(pos)
        node_pos.appendChild(doc.createTextNode(str(obj_dict['box'][pos])))
        node_box.appendChild(node_pos)

    object_node.appendChild(node_truncated)
    object_node.appendChild(node_difficult)
    object_node.appendChild(node_box)

    return object_node


# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    sys.exit(1)

frames_to_be_inspected = 100

frame_timings = list(range(0, num_frames, int(math.ceil(num_frames/frames_to_be_inspected))))
frame_table = [None]*frames_to_be_inspected
frameC = 0
for frameIdx, (t, frame) in enumerate(video.iterframes(with_time=True)):
    if frameIdx in frame_timings:
        frame_table[frameC] = (frameIdx, t)
        print(frameC, frameIdx, t)
        frameC += 1        

currentFrame = 0

doc = None

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_table[currentFrame][0])
    except IndexError:
        print('Congrats!')
        sys.exit(0)
    print('Current frame: ', currentFrame)    
    ret, frame = cap.read()
    code = None
    doc = generate_new_doc(video_descriptor.split('.')[0], frame_table[currentFrame][1], frame_table[currentFrame][0],
                         cap_width, cap_height, 3)
    if ret:

        # q: quit
        # b: one frame backward
        # d: dense region box
        # ENTER: car box
        # c: one frame forward
        # r: regret - delete last box

        # g: generate blank object list
        # s: save the doc

        objects = []

        while True:
            # Display the resulting frame
            frame_annotated = frame.copy()
            for item in objects:
                # draw car
                if item['type'] == 'car':
                    print('draw car')
                    cv2.rectangle(frame_annotated,
                                  (item['box'][0], item['box'][1]),
                                  (item['box'][2], item['box'][3]), (0, 255, 0))
                # draw ignored
                else:
                    print('draw ignored')
                    cv2.rectangle(frame_annotated, (xmin, ymin), (xmax, ymax), (0, 0, 0), -1)
            cv2.imshow('Frame', frame_annotated)

            # Wait for user response
            key = cv2.waitKey(0) & 0xff # ord('q'):

            if key == ord('q'):
                code = 'q'
                break
            elif key == ord('g'):
                objects = []
                print('New list generated')
                break
            elif key == ord('r'):
                objects = objects[:-1]
                print('Removed last box')
                continue
            elif key in [13, ord('d')]:
                # Select ROI
                fromCenter = False
                showCrosshair = False

                x, y, w, h = cv2.selectROI("Frame", frame_annotated, fromCenter, showCrosshair)
                xmin, ymin, xmax, ymax = x, y, x+w, y+h
                if not (xmin, ymin, xmax, ymax) == (0, 0, 0, 0):
                    if key == 13:
                        objects.append({
                                'type': 'car',
                                'box':  (xmin, ymin, xmax, ymax)})
                    else:
                        objects.append({
                                'type': 'ignored',
                                'box': (xmin, ymin, xmax, ymax)})
                continue
            elif key == ord('c'):
                numFrameToSkip = int(6*fps)
                currentFrame += 1
                print('next frame: {}'.format(currentFrame) + '. Continue\n')
                break
            elif key == ord('s'):

                # Do not save a file with no box in it.
                if len(objects) == 0:
                    break


                cars = []
                ignored = []
                root = doc.getElementsByTagName('annotation')[0]
                for item in objects:
                    if item['type'] == 'car':
                        root.appendChild(prepareObj(item))
                    else:
                        root.appendChild(prepareObj(item))

                # sanity check: display the boxes in the image
                # img = cv2.cvtColor(video._get_frame(frame_table[currentFrame][1]), cv2.COLOR_RGB2BGR)

                # objects = doc.getElementsByTagName('object')
                # ignored = doc.getElementsByTagName('ignored')
                # for obj in objects + ignored:
                #    box = obj.getElementsByTagName('box')[0]
                #    name = 'car' if obj in objects else 'ignored'
                #    xmin, ymin, xmax, ymax = map(lambda x: int(box.getElementsByTagName(x)[0].firstChild.nodeValue), ['xmin', 'ymin', 'xmax', 'ymax'])
                #    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0) if name == 'car' else (0, 0, 255))
                # cv2.imshow('confirm annotation', img)
                # confirm = cv2.waitKey(0) & 0xff
                # if confirm == ord('y'):


                # Write XML
                with open('{:03d}'.format(int(video_descriptor.split('.')[0])) + '_' + '{:03d}'.format(currentFrame+1) + '.xml', 'w') as f:
                    doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")
                    print('XML file saved.')
                    continue
                    # doc = generateNewDoc(videoDescriptor, frame_table[currentFrame][1],
                    #                     frame_table[currentFrame][0], width, height, 3)
                # else:
                #     print('XML file not saved.')
                # cv2.destroyWindow('confirm annotation')
            elif key == ord('b'):
                if currentFrame > 0:
                    currentFrame -= 1
                print('back to frame: {}'.format(currentFrame) + '. Continue\n')
                break
        else:
            print('press q to quit, b to box, and c to continue')
            continue
        if code == 'q':
            break
    # Break the loop
    else:
        break



# When everything done, release the video capture obj
cap.release()

# Closes all the frames
cv2.destroyAllWindows()



