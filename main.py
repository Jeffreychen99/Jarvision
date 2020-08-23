import time
import numpy as np
import cv2
import face_recognition
from mss import mss
from PIL import Image
from threading import Thread
from queue import Queue 


SCALE_FACTOR = 2
SCREEN_SIZE = (1230, 800)


def drawEye(frame, eye_coords):
	for pt in eye_coords:
		cv2.circle(frame, pt, 1, (0, 255, 0), 1)

def findEyeCenter(eye_frame, display=False):
	gray_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
	blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
	_, filtered_eyes = cv2.threshold(blurred_frame, gray_frame.mean()*0.425, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(filtered_eyes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]

	if display:
		cv2.drawContours(eye_frame, [contours[0]], -1, (255, 0, 0), 1)

	M = cv2.moments(contours[0])

	cx, cy = 0, 0
	try:
		cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
		if display:
			cv2.circle(eye_frame, (cx, cy), 1, (0, 0, 255), 2)
	except:
		pass

	return (cx, cy)

def getEyeFrameBounds(frame, eye_points):
	eye_bounds = {
		'left': int( eye_points[0][0] * 0.95 ),
		'right': int( eye_points[3][0] * 1.05 ),
		'top': int( min(eye_points, key=lambda p: p[1])[1] * 0.95 ),
		'bot': int( max(eye_points, key=lambda p: p[1])[1] * 1.05 )
	}

	return eye_bounds

def packageEyeData(left_eye, right_eye, eye_centers):
	def findFrame(eye):
		left_bound = min(eye[1][0], eye[5][0])
		right_bound = min(eye[2][0], eye[4][0])

		upper_bound = min(eye[1][1], eye[2][1])
		lower_bound = min(eye[5][1], eye[4][1])

		return ((left_bound, upper_bound), (right_bound, lower_bound))

	return (findFrame(left_eye), eye_centers[0]), (findFrame(right_eye), eye_centers[1])



def captureWebCamStream(queue, display=False):
	camera_cap = cv2.VideoCapture(0)

	# Only need to process every other frame (for speed purposes)
	process_this_frame = True

	while True:
		ret, frame = camera_cap.read()
		frame = np.array(frame[:, -1::-1, :])

		# Resize frame of video to 1/4 size for faster face recognition processing
		small_frame = cv2.resize(frame, (0, 0), fx=1/SCALE_FACTOR, fy=1/SCALE_FACTOR)

		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:, :, ::-1]

		# Only need to process every other frame (for speed purposes)
		if process_this_frame:
			landmarks_ls = face_recognition.face_landmarks(rgb_small_frame)
		
		# Display the results
		for landmarks in landmarks_ls:
			left_eye_points = [(p[0] * SCALE_FACTOR, p[1] * SCALE_FACTOR) for p in landmarks['left_eye']]
			right_eye_points = [(p[0] * SCALE_FACTOR, p[1] * SCALE_FACTOR) for p in landmarks['right_eye']]

			l_bounds = getEyeFrameBounds(frame, left_eye_points)
			r_bounds = getEyeFrameBounds(frame, right_eye_points)

			left_eye_frame = frame[ l_bounds['top']:l_bounds['bot'], l_bounds['left']:l_bounds['right'], :]
			right_eye_frame = frame[ r_bounds['top']:r_bounds['bot'], r_bounds['left']:r_bounds['right'], :]

			left_eye_center = findEyeCenter(left_eye_frame, display=display)
			right_eye_center = findEyeCenter(right_eye_frame, display=display)

			left_eye_center = tuple( np.add(left_eye_center, (l_bounds['left'], l_bounds['top'])) )
			right_eye_center = tuple( np.add(right_eye_center, (r_bounds['left'], r_bounds['top'])) )

			eye_data = packageEyeData(left_eye_points, right_eye_points, [left_eye_center, right_eye_center])
			queue.put(eye_data)

			look_point = processEyeData(frame, eye_data)
			cv2.circle(frame, (look_point[0], len(frame)//2), 3, (255, 0, 255), 3)

			if display:
				drawEye(frame, left_eye_points)
				drawEye(frame, right_eye_points)
				cv2.imshow('left_eye_frame', left_eye_frame)
				cv2.imshow('right_eye_frame', right_eye_frame)


		if display:
			# Display the resulting image
			cv2.imshow('Video', frame)

			# Hit 'q' on the keyboard to quit
			if cv2.waitKey(1) & 0xFF == ord('q'):
				camera_cap.release()
				break
		



def processEyeData(frame, eye_data):
	left_data = eye_data[0]
	right_data = eye_data[1]

	left_center_x = (left_data[1][0] - left_data[0][0][0]) / (left_data[0][1][0] - left_data[0][0][0])
	right_center_x = (right_data[1][0] - right_data[0][0][0]) / (right_data[0][1][0] - right_data[0][0][0])

	left_center_y = (left_data[1][1] - left_data[0][0][1]) / (left_data[0][1][1] - left_data[0][0][1])
	right_center_y = (right_data[1][1] - right_data[0][0][1]) / (right_data[0][1][1] - right_data[0][0][1])

	avg_center_x = (left_center_x + right_center_x) / 2
	avg_center_y = (left_center_y + right_center_y) / 2

	h, w, _ = frame.shape
	look_point = ( int(avg_center_x * w), int(avg_center_y * h) )

	return look_point

def screenStream(queue, display=False):
	sct = mss()
	w, h = 500, 1000

	centroid_num = 5
	centroid_point = None
	while True:
		screen_bounds = {'top': 0, 'left': 0, 'width': SCREEN_SIZE[0], 'height': SCREEN_SIZE[1]}
		#screen_frame = np.array( sct.grab(screen_bounds) )

		#small_frame = cv2.resize(screen_frame, (0, 0), fx=0.35, fy=0.35)

		small_frame = np.zeros((w, h, 3))

		eye_data = queue.get()
		look_point = processEyeData(small_frame, eye_data)
		if not centroid_point:
			centroid_point = look_point
		else:
			centroid_point = np.add( np.array(centroid_point) * (centroid_num - 1), look_point) // centroid_num
			centroid_point = tuple(centroid_point)

		cv2.circle(small_frame, centroid_point, 3, (255, 0, 255), 3)

		if display:
			cv2.imshow('screen', small_frame)

		if (cv2.waitKey(1) & 0xFF) == ord('q'):
			break


if __name__ == '__main__':

	inter_thread_queue = Queue()

	captureWebCamStream(inter_thread_queue, display=True)
	#web_cam_thread = Thread(target=captureWebCamStream, args=(inter_thread_queue,))
	#web_cam_thread.start()

	#screenStream(inter_thread_queue, display=True)

	cv2.destroyAllWindows()