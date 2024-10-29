import yt_dlp
import whisper
import ffmpeg
import torch
from datetime import timedelta
from tqdm import tqdm 

def download_youtube_video(youtube_url, output_path="youtube_video.mp4"):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path


def extract_audio(video_path, audio_path="audio.aac"):
    ffmpeg.input(video_path).output(audio_path, vn=None).run()
    return audio_path


def transcribe_audio(audio_path, model_size="small"):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Використання GPU, якщо доступний
    model = whisper.load_model(model_size, device=device)

    result = model.transcribe(audio_path, language="uk", verbose=False)
    segments = result['segments']

    print("Транскрипція аудіо:")
    for segment in tqdm(segments, desc="Обробка сегментів"):
        pass  # Це дозволяє `tqdm` оновлювати індикатор прогресу для кожного сегмента
    return segments


def format_transcription_to_notes(segments):
    notes = []
    for segment in segments:
        start_time = str(timedelta(seconds=int(segment['start'])))
        end_time = str(timedelta(seconds=int(segment['end'])))
        text = segment['text']

        notes.append(f"## Час: {start_time} - {end_time}")
        notes.append(text + "\n")
    return "\n".join(notes)


def capture_screenshots(video_path, timestamps):
    screenshots = []
    for timestamp in timestamps:
        formatted_timestamp = timestamp.replace(":", "-")
        output_filename = f"screenshot_{formatted_timestamp}.png"
        (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output(output_filename, vframes=1)  
            .run()
        )
        screenshots.append(output_filename)
    return screenshots

def create_notes_from_youtube_video(youtube_url, model_size="small"):

    video_path = download_youtube_video(youtube_url)

    audio_path = extract_audio(video_path)

    segments = transcribe_audio(audio_path, model_size=model_size)

    notes = format_transcription_to_notes(segments)

    timestamps = [str(timedelta(seconds=i * 300)) for i in range(len(segments) // 10)]
    screenshots = capture_screenshots(video_path, timestamps)

    for idx, screenshot in enumerate(screenshots):
        notes += f"\n![Скріншот {idx + 1}](screenshot_{timestamps[idx].replace(':', '-')}.png)\n"

    with open("lecture_notes.md", "w", encoding="utf-8") as file:
        file.write(notes)

    print("Конспект успішно створено та збережено у 'lecture_notes.md'")


create_notes_from_youtube_video(
    "https://www.youtube.com/watch?v=h12JVB-wcoQ",
    model_size="small" 
)
