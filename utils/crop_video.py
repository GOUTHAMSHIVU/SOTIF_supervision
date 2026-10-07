from moviepy.video.io.VideoFileClip import VideoFileClip

# 1. Load the original video file
# Replace 'input_video.mp4' with the path to your actual video
video_path = r"D:\SOTIF_supervision_workspace\data\ood\videoplayback.mp4"
output_path = r"D:\SOTIF_supervision_workspace\data\ood\videoplayback_cropped.mp4"

# 2. Open the video clip and crop the timeline
# You can use "MM:SS" strings or total seconds: 3:00 = 180s, 3:45 = 225s
with VideoFileClip(video_path) as video:
    cropped_clip = video.subclipped("00:00:36", "00:00:50")
    
    # 3. Write the result to a new file
    # 'codec' ensures high compatibility, 'audio_codec' preserves the sound
    cropped_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac"
    )

print("Video successfully cropped!")
