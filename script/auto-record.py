import cv2
import sys
import numpy as np
import os
import time
import math
import datetime

def define_stream_capture(stream_url):
	vcap   = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
	fps_in = math.ceil(vcap.get(cv2.CAP_PROP_FPS))
	return vcap, fps_in

def display_message(mes):
	sys.stdout.write("\033[F")
	print(mes)

def record_stream(period, stream_url, output_video, output_image):
	# Inite parameter
	vcap   = None
	vwrite = None
	video_ou_h, video_ou_w  = None, None
	vwrite = None
	age_one_stream = 300  # second(s)

	# Init the information
	os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
	vcap, fps_in = define_stream_capture(stream_url)
	# cv2.namedWindow("STREAM", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions


	# Annouce the information
	print(f"Stream : {stream_url}")
	print(f"Output video : {output_video}")
	print(f"Frames per second of stream : {fps_in}")
	print(f"Delay period {period} seconds")
	print(f"Start record at ::  {datetime.datetime.now()}")
	print()
	
	# Recording
	period_tmp = period * fps_in
	age_wait   = 0
	num_frame  = 0
	while(1):
		period_tmp += 1

		# Wait until the period
		if period_tmp < period * fps_in:
			ret = vcap.grab()
			continue
		else:
			ret, frame = vcap.read()
		
		# process stream
		if ret == False:
			display_message("Frame is empty")
			period_tmp -= 1
			age_wait   += 1

			if age_wait >= age_one_stream * fps_in:
				age_wait = 0

				# create new stream
				vcap.release()
				vcap, fps_in = define_stream_capture(stream_url)
				print("Define the New stream")
				print()

			continue
			# break
		else:
			# reset period
			period_tmp = 0
			age_wait   = 0

			# Init video
			if vwrite is None:  # if video is not initiated
				video_ou_h, video_ou_w, _ = frame.shape
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				fps_ou = 2
				vwrite = cv2.VideoWriter(output_video, fourcc, fps_ou, (video_ou_w, video_ou_h))

			# Write video
			num_frame += 1
			vwrite.write(frame)

			# Display
			display_message(f"Current time :: {datetime.datetime.now()} -- Number of frame: {num_frame}")
			# cv2.imshow('STREAM', frame)
			cv2.imwrite(output_image, frame)
			if cv2.waitKey(20) & 0xFF == 27:
				# print(f"Current time : {datetime.datetime.now()} -- Number of frame: {num_frame}")
				break

	# Post process
	cv2.destroyAllWindows()
	if vcap is not None:
		vcap.release()
	if vwrite is not None:
		vwrite.release()
	print(f"Stop record at ::  {datetime.datetime.now()}")


if _name_ == "__main__":
	period        = 15  # second(s) for 
	index_stream  = 70
	ip_adr        = "211.34.147.236"
	stream_video  = f"rtsp://{ip_adr}:2935/live/{index_stream}.stream"
	output_folder = "/media/sugarmini/Data1/2_dataset/CCTV_record/timelapse/"
	output_video  = f"{ip_adr}_2935_live_{index_stream}_pe_{period}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.mp4"
	output_image  = f"{ip_adr}_2935_live_{index_stream}_pe_{period}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.png"
	record_stream(period, stream_video, os.path.join(output_folder, output_video), os.path.join(output_folder,output_image))