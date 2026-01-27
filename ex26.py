import cv2

def reverse_video(input_path,output_path):
	video=cv2.VideoCapture(input_path)
	if not video.isOpened():
		print("Error: Cannot open video file.")
		return
	frame_width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps=video.get(cv2.CAP_PROP_FPS)
	frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	frames=[]
	for i in range(frame_count):
		ret,frame=video.read()
		if not ret:
			break
		frames.append(frame)
	video.release()
	out=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame_width,frame_height))
	for frame in reversed(frames):
		out.write(frame)
	out.release()
	print("Video reversed successfully!")

# Example usage
reverse_video(r"D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\samplev.mp4",
              r"D:\SIMATS Engineering\ITA0513 Computer Vision\Practical\sources\samplev_reversed.mp4")
