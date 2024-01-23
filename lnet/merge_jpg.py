import subprocess

def images_to_video(image_folder, output_video, fps):
    command = f"ffmpeg -framerate {fps} -i {image_folder}/%d.jpg -c:v libx264 -pix_fmt yuv420p {output_video}"
    subprocess.run(command, shell=True)

# 使用函数
images_to_video('/datadisk1/zhouxc/CY/Lips/video_retalk/lnet/_lpip_save_/train_save/IMG/2024-01-10_11-09-28', 'output_video.mp4', 25)