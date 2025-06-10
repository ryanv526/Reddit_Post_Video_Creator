# Reddit_Post_Video_Creator
Reddit Subtitle Generator
This Python script automates the creation of YouTube Shorts-style videos featuring animated subtitles, synchronized with text-to-speech (TTS) audio. It's designed to bring Reddit stories to life by combining background videos with dynamic, word-timed captions, an introductory title card, and optional word obfuscation.

It leverages Whisper AI for highly accurate word timing, Amazon Polly for natural-sounding voices, and MoviePy for efficient video composition.

Features
Automated Subtitles: Generates animated, word-by-word subtitles.

Accurate Timing: Utilizes Whisper AI for precise synchronization of subtitles with speech. Falls back to an intelligent estimation method if Whisper is unavailable or struggles.

Amazon Polly TTS: Converts story text into natural-sounding speech using Amazon Polly's Neural voices (requires AWS credentials). Handles long texts by chunking them to respect Polly's character limits.

Customizable Intro Card: Features an introductory title card displaying the story's title, author, a "rewards" image, and a randomly selected Reddit avatar.

Abbreviation Expansion: Automatically expands common internet and Reddit abbreviations (e.g., AITA, SAHM) for clearer pronunciation in TTS.

Word Obfuscation: Allows for sensitive words to be replaced with pre-defined alternatives (e.g., swear words) in the displayed subtitles.

High-Quality Output: Renders vertical videos suitable for platforms like YouTube Shorts, TikTok, and Instagram Reels.

Clean for GitHub: No hardcoded sensitive API keys or local file paths. All external resources are configured via environment variables or command-line arguments.

  Prerequisites
Before you can run this script, ensure you have the following installed:

Python 3.x: (Tested with Python 3.8+)

FFmpeg: An open-source multimedia framework that MoviePy depends on.

Python Libraries: moviepy, pydub, boto3, openai-whisper, torch.

AWS Account & Credentials: For Amazon Polly TTS.

🚀 Installation & Setup
Follow these steps to set up the project on your machine.

1. Install Python 3
If you don't have Python 3, download it from python.org.
Crucially, during installation, ensure you check "Add Python X.X to PATH".

2. Install FFmpeg
Download: Go to Gyan's FFmpeg Builds. Download ffmpeg-master-latest-win64-gpl.zip. This is a static build, which is self-contained and generally easiest for Windows users.

Extract: Create a new folder (e.g., C:\ffmpeg) and extract the contents of the downloaded zip file into it. You should find a bin folder inside the extracted structure (e.g., C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin).

Add to PATH:

Search for "Environment Variables" in Windows search and select "Edit the system environment variables."

Click "Environment Variables..."

Under "System variables," find and select Path, then click "Edit."

Click "New" and add the full path to the bin folder where ffmpeg.exe is located (e.g., C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin).

Click "OK" on all windows to save.

Verify: Open a new Command Prompt/PowerShell window and type ffmpeg -version. You should see version details.

3. Install Python Libraries
Open your terminal or command prompt and run:

pip install moviepy pydub boto3 "openai-whisper" "torch"

4. Configure AWS Credentials
The script uses boto3 to access Amazon Polly. For security, do NOT hardcode your AWS credentials in the script for GitHub. Instead, configure them securely:

Recommended Method (AWS CLI):

Install the AWS CLI.

Open your terminal and run aws configure.

Enter your AWS Access Key ID, AWS Secret Access Key, and Default region name (e.g., us-west-2 as specified in the script's default region).

Where to get these: Log into your AWS Management Console. Go to IAM -> Users -> (Your User Name) -> Security credentials tab. You can create new access keys there.

Alternative Method (Environment Variables):
Set these environment variables on your system:

AWS_ACCESS_KEY_ID

AWS_SECRET_ACCESS_KEY

AWS_REGION (e.g., us-west-2)

5. Prepare Project Files
Organize your project files as follows:

your_project_folder/
├── reddit_subtitles_fixed.py
├── story.json               # Your Reddit story data
├── obfuscation.json         # (Optional) Words to replace
├── rewards.png              # Image for the intro card
├── RedditAvatars/           # Folder containing various .png avatar images
│   ├── avatar1.png
│   ├── avatar2.png
│   └── ...
└── background_video.mp4     # Your background footage

story.json Example:

{
  "story_title": "My Incredible Journey through Reddit Confessions",
  "post_author": "u/AnonymousReader99",
  "story_text": "This is the very long story text that will be converted to speech and displayed as subtitles. It can contain multiple paragraphs and sentences. For example, AITA for telling my SAHM sister that her children are spoiled? She got really upset, saying AITB for not understanding the struggles of a SAHD. TLDR: arguments, arguments, and more arguments. LOL. BTW, I also had to deal with a lot of FOMO from friends about my IRL experiences, and SMH, it just kept getting worse. Anyway, that's my story for today!"
}

Important: Ensure all double quotes (") within string values in your JSON are escaped with a backslash (\"). Typographic quotes (“ ”) do not need escaping.

obfuscation.json Example:

{
  "hell": ["heck", "h-e-double-hockey-sticks"],
}

The keys are the words to obfuscate (lowercase), and values are a list of replacement options.

Fonts: The script defaults to looking for Montserrat-SemiBold.ttf and Arial.ttf in common system font directories or the current working directory. If you want to use specific fonts, provide their full paths using the --font-main-path and --font-fallback-path arguments.

🚀 Usage
Navigate to your your_project_folder in the terminal and run the script with the required arguments.

cd path/to/your_project_folder

Basic Command (using default file paths for story.json, obfuscation.json, rewards.png, and RedditAvatars):

python reddit_subtitles_fixed.py "path/to/your/background_video.mp4" "path/to/your/output_video.mp4"

Full Command (specifying all optional arguments and paths):

python reddit_subtitles_fixed.py \
    "C:\Users\YourName\Videos\MyBackgroundFootage.mp4" \
    "C:\Users\YourName\Desktop\MyAwesomeShort.mp4" \
    --story-json "C:\Users\YourName\Documents\my_custom_story.json" \
    --obfuscation-json "C:\Users\YourName\Configs\my_obfuscation_rules.json" \
    --rewards-img "C:\Users\YourName\Assets\special_rewards_icon.png" \
    --reddit-avatars-folder "C:\Users\YourName\MyAvatarsFolder" \
    --font-main-path "C:\Windows\Fonts\Montserrat-Bold.ttf" \
    --font-fallback-path "C:\Windows\Fonts\Arial.ttf" \
    --voice-gender M \
    --force-estimation

Command-Line Arguments:
Positional Arguments (Required):

background_video: Path to the input background video file (e.g., assets/background.mp4).

output: Desired path for the output video file (e.g., output/final_video.mp4).

Optional Arguments:

--story-json TEXT: Path to the JSON file containing story_title, post_author, and story_text. Defaults to story.json.

--obfuscation-json TEXT: Path to the JSON file for word obfuscation. Defaults to obfuscation.json.

--rewards-img TEXT: Path to the rewards image file (.png) for the intro card. Defaults to rewards.png.

--reddit-avatars-folder TEXT: Path to the folder containing Reddit avatar PNGs. Defaults to RedditAvatars.

--font-main-path TEXT: Path to the main font file (e.g., Montserrat-SemiBold.ttf). Script will search common system paths if not found.

--font-fallback-path TEXT: Path to a fallback font file (e.g., Arial.ttf). Script will search common system paths if not found.

--force-estimation: If set, skips Whisper AI analysis and uses only the faster (but less accurate) estimation method for word timing.

--voice-gender [J|M]: Choose voice gender for Amazon Polly. J for Joanna (female, default), M for Matthew (male).

  Troubleshooting
ffmpeg not found: Ensure FFmpeg is installed and its bin directory is added to your system's PATH environment variable.

json.JSONDecodeError: Check your story.json or obfuscation.json for syntax errors. Common issues include missing commas, unclosed brackets/braces, or unescaped double quotes (") within string values (use \" instead).

TextLengthExceededException from Amazon Polly: This is handled by the script, which now automatically chunks long texts. If you still encounter this, ensure your POLLY_MAX_CHARS in the script is not set higher than 3000 for neural voices (2950 is used by default for safety).

Slow Rendering: The script uses preset="fast" for rendering. If it's still too slow, you can try changing the preset in the write_videofile call within the create_subtitle_video function to superfast or ultrafast for faster but potentially lower-quality output.

No AWS Credentials Error: Make sure your AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION environment variables are set correctly, or that you've configured the AWS CLI via aws configure.

  License
This project is open-source and available under the MIT License.
