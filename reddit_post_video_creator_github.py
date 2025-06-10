#!/usr/bin/env python3
"""
Reddit TTS Subtitle Generator - Whisper Integration
Creates YouTube Shorts-style animated subtitles synced with TTS audio.
Uses Whisper AI for maximum accuracy word timing, and Amazon Polly for voice.

This version is designed for GitHub, with no hardcoded AWS credentials or
local file paths. All paths are provided via command-line arguments.
"""

import os
import sys
import argparse
import re
import wave
import contextlib
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, VideoClip, CompositeVideoClip, TextClip, CompositeAudioClip
import moviepy.video.fx.all as fx
from PIL import Image, ImageDraw, ImageFont
import tempfile
import json
import shutil
import random

# Import Amazon Polly client
try:
    import boto3
    POLLY_AVAILABLE = True
    print("✅ Amazon Polly client (boto3) available.")
except ImportError:
    POLLY_AVAILABLE = False
    print("❌ Amazon Polly client (boto3) not available.")
    print("Install with: pip install boto3")


# Whisper import with fallback
try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
    print("✅ Whisper AI available for maximum accuracy")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper AI not available, using fallback methods")
    print("For best accuracy, install with: pip install pip install openai-whisper torch")

class RedditTTSSubtitles:
    # Define a maximum character limit for Amazon Polly's synthesize_speech operation.
    # Neural voices typically support up to 3000 characters. Using a slightly lower
    # value for safety and to allow for chunking at natural breaks.
    POLLY_MAX_CHARS = 2950 # Adjusted for neural voices (max 3000)

    def __init__(self, font_main_path, font_fallback_path, reddit_avatars_folder, obfuscation_file_path):
        self.temp_dir = tempfile.mkdtemp()
        self.whisper_model = None

        # Amazon Polly TTS client and availability
        self.polly_client = None
        self.polly_available = False

        # AWS Credentials are now loaded automatically by boto3 from environment variables,
        # AWS shared credentials file (~/.aws/credentials), or IAM roles.
        # DO NOT hardcode them here for GitHub.
        if POLLY_AVAILABLE:
            try:
                # boto3 will automatically pick up credentials from environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
                # or from ~/.aws/credentials and ~/.aws/config files.
                self.polly_client = boto3.client('polly') 
                self.polly_available = True
                print("✅ Amazon Polly client initialized. Credentials are loaded from environment/config.")
            except Exception as e:
                print(f"❌ Error initializing Amazon Polly client. Ensure AWS credentials are configured: {e}")
                print("   (e.g., via environment variables like AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, or AWS CLI 'aws configure').")
                self.polly_available = False # Explicitly set to False on error
        else:
            print("❌ Amazon Polly module (boto3) not loaded due to ImportError.")

        # Font paths - now passed as arguments to __init__
        self.font_path_main = font_main_path
        self.font_path_fallback = font_fallback_path

        # Path to Reddit Avatars folder - now passed as argument
        self.reddit_avatars_folder = reddit_avatars_folder

        # Load Whisper model if available
        if WHISPER_AVAILABLE:
            try:
                print("Loading Whisper model (this may take a moment first time)...")
                # You might consider loading a larger model like "small" or "medium" for better accuracy
                # self.whisper_model = whisper.load_model("small")
                self.whisper_model = whisper.load_model("base")
                print("✅ Whisper model loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load Whisper model: {e}")
                self.whisper_model = None

        # Dictionary for common abbreviations to full phrases
        self.abbreviation_map = {
            "IDK": "I don't know",
            "BTW": "by the way",
            "LOL": "LOL",          # Keep as is
            "BRB": "BRB",          # Keep as is
            "OMG": "OMG",          # Keep as is
            "ASAP": "as soon as possible",
            "FAQ": "frequently asked questions",
            "FYI": "for your information",
            "NVM": "never mind",
            "TMI": "too much information",
            "IMO": "in my opinion",
            "IMHO": "in my humble opinion",
            "TLDR": "TLDR",        # Keep as is
            "AFAIK": "as far as I know",
            "FOMO": "FOMO",        # Keep as is
            "IRL": "IRL",          # Keep as is
            "ROFL": "ROFL",        # Keep as is
            "SMH": "shaking my head",
            "TBD": "to be determined",
            "AKA": "also known as",
            "DIY": "do it yourself",
            "ETA": "estimated time of arrival",
            "EOD": "end of day",
            "COB": "close of business",
            "OT": "off topic",
            "WFH": "work from home",
            "TTYL": "talk to you later",
            "G2G": "got to go",
            "IDC": "I don't care",
            "THX": "thanks",
            "NP": "no problem",
            "JK": "just kidding",
            "LMK": "let me know",
            "P.S.": "postscript",
            "RSVP": "RSVP",        # Keep as is
            "E.G.": "for example",
            "I.E.": "that is",
            "VS.": "versus",
            "C.V.": "C.V.",        # Keep as is
            "F.Y.I.": "for your information", # with dots
            "S.M.H.": "shaking my head", # with dots
            "AITB": "Am I the Butthole",
            "AITA": "Am I the Asshole",
            "SAHM": "Stay at Home Mom",
            "SAHD": "Stay at home Dad",
        }

        self.obfuscation_map = {}
        # Obfuscation file path is now passed as an argument
        try:
            if obfuscation_file_path and os.path.exists(obfuscation_file_path):
                with open(obfuscation_file_path, 'r', encoding='utf-8') as f:
                    self.obfuscation_map = json.load(f)
                print(f"✅ Loaded obfuscation map from: {obfuscation_file_path}")
            else:
                print(f"⚠️ Obfuscation file not found at: {obfuscation_file_path if obfuscation_file_path else 'None provided'}. No words will be obfuscated.")
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format in obfuscation file: {obfuscation_file_path}. No words will be obfuscated.")
            self.obfuscation_map = {}
        except Exception as e:
            print(f"❌ Error loading obfuscation map: {e}. No words will be obfuscated.")
            self.obfuscation_map = {}


    def clean_text(self, text):
        """Clean and prepare text for TTS, including expanding common abbreviations."""
        # Remove Reddit formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)*', r'\1', text)      # Italic
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'\n+', ' ', text)              # Multiple newlines
        text = re.sub(r'\s+', ' ', text).strip()      # Multiple spaces

        # Expand common abbreviations (case-insensitive search, replace preserves case if possible)
        # Iterate over sorted keys to ensure longer phrases are matched before their sub-parts
        sorted_keys = sorted(self.abbreviation_map.keys(), key=len, reverse=True)
        for abbr in sorted_keys:
            # Use regex with word boundaries to avoid replacing parts of other words
            # re.escape handles special characters in keys like '.'
            pattern = r'\b' + re.escape(abbr) + r'\b'
            replacement = self.abbreviation_map[abbr]
            # Replace case-insensitively using re.IGNORECASE
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _obfuscate_word(self, word):
        """Obfuscates a word if it's in the obfuscation map."""
        word_lower = word.lower()
        if word_lower in self.obfuscation_map:
            obfuscated_options = self.obfuscation_map[word_lower]
            if obfuscated_options:
                return random.choice(obfuscated_options)
        return word # Return original word if not found or no options

    def get_whisper_word_timings(self, audio_path, original_text):
        """Use Whisper AI for precise word-level timestamps"""
        if not self.whisper_model:
            return None

        print("🎯 Analyzing audio with Whisper AI for precise word timing...")

        try:
            # Transcribe with word-level timestamps
            result = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                language='en',
                verbose=False
            )

            # Extract word-level timings
            word_timings = []

            for segment in result['segments']:
                for word_info in segment.get('words', []):
                    word_timings.append({
                        'word': word_info['word'].strip(),
                        'start': word_info['start'],
                        'end': word_info['end'],
                        'duration': word_info['end'] - word_info['start'],
                        'confidence': word_info.get('probability', 1.0)
                    })

            print(f"✅ Whisper found {len(word_timings)} words with precise timestamps")

            # Quality check - if Whisper found too few words, use hybrid approach
            original_words = original_text.split()
            # This threshold might need fine-tuning based on your content/model
            if len(word_timings) < len(original_words) * 0.7:  # Less than 70% match
                print("🔄 Whisper missed some words, using hybrid approach...")
                return self.create_hybrid_timings(word_timings, original_words, audio_path)

            return word_timings

        except Exception as e:
            print(f"❌ Whisper analysis failed: {e}")
            return None

    def create_hybrid_timings(self, whisper_timings, original_words, audio_path):
        """Combine Whisper results with intelligent gap filling"""
        print("🔧 Creating hybrid timing with gap filling...")

        # Get total audio duration
        try:
            audio_clip = AudioFileClip(audio_path)
            total_duration = audio_clip.duration
            audio_clip.close()
        except:
            total_duration = len(original_words) * 0.4 # Fallback if audio duration can't be read

        hybrid_timings = []
        whisper_idx = 0

        for i, original_word in enumerate(original_words):
            # Try to find matching word in Whisper results
            found_match = False

            # Look for word match in next few Whisper results (sliding window)
            for j in range(whisper_idx, min(whisper_idx + 5, len(whisper_timings))): # Increased search window
                whisper_word = whisper_timings[j]['word'].lower().strip('.,!?\'"') # More robust cleaning
                original_clean = original_word.lower().strip('.,!?\'"')

                # Check for exact match or contains match (case-insensitive)
                if (whisper_word == original_clean or
                    (len(original_clean) > 2 and original_clean in whisper_word) or
                    (len(whisper_word) > 2 and whisper_word in original_clean)):

                    # Use Whisper timing but with original word (cleaner capitalization, etc.)
                    timing = whisper_timings[j].copy()
                    timing['word'] = original_word
                    hybrid_timings.append(timing)
                    whisper_idx = j + 1 # Advance whisper index
                    found_match = True
                    break

            if not found_match:
                # Estimate timing for missing word
                if hybrid_timings:
                    last_end = hybrid_timings[-1]['end']
                    # Estimate based on word complexity
                    word_duration = self.estimate_word_duration(original_word)
                    # Add small pause based on punctuation
                    pause = 0.15 if original_word.endswith(('.', '!', '?')) else 0.05
                    start_time = last_end + pause
                else:
                    # First word - estimate position (initial small delay)
                    start_time = 0.1
                    word_duration = self.estimate_word_duration(original_word)

                hybrid_timings.append({
                    'word': original_word,
                    'start': start_time,
                    'end': start_time + word_duration,
                    'duration': word_duration,
                    'confidence': 0.4  # Lower confidence for estimation
                })

        # Ensure that the last word's end time is not beyond the audio duration (with a small buffer)
        if hybrid_timings and hybrid_timings[-1]['end'] > total_duration + 0.1:
             hybrid_timings[-1]['end'] = total_duration # Cap the end time if it overshoots

        print(f"✅ Hybrid timing created: {len(hybrid_timings)} words")
        return hybrid_timings

    def estimate_word_duration(self, word):
        """Estimate word duration based on complexity"""
        base_duration = 0.3

        # Length factor
        length_factor = len(word) * 0.02 # Slightly reduced impact of length

        # Complexity factor (simplified for basic estimation)
        complexity = 0
        vowels = 'aeiouAEIOU'
        syllables = sum(1 for char in word if char in vowels)
        if syllables > 2:
            complexity += (syllables - 2) * 0.05

        return base_duration + length_factor + complexity

    def analyze_speech_timing(self, audio_path, text):
        """Master function - tries Whisper first, falls back to other methods"""
        print("🎯 Starting speech timing analysis...")

        # Try Whisper first (most accurate)
        if self.whisper_model:
            whisper_result = self.get_whisper_word_timings(audio_path, text)
            if whisper_result:
                return whisper_result

        # Fallback to estimation
        print("🔄 Falling back to estimation method...")
        try:
            # Get audio duration
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            audio_clip.close()
        except:
            duration = len(text.split()) * 0.4 # Very rough estimate

        return self.estimate_word_timings(text, duration)

    def estimate_word_timings(self, text, audio_duration):
        """Enhanced estimation with better speech patterns"""
        print("Using enhanced word timing estimation...")
        words = text.split()
        timings = []

        # Calculate total estimated complexity score (for scaling)
        total_estimated_duration = sum(self.estimate_word_duration(word) for word in words)
        
        current_time = 0.1  # Small initial delay

        for i, word in enumerate(words):
            word_duration = self.estimate_word_duration(word)
            
            # Scale word duration to fit the overall audio duration, leaving some room for pauses
            if total_estimated_duration > 0: # Avoid division by zero
                scaled_duration = (word_duration / total_estimated_duration) * audio_duration * 0.9 # 10% for pauses
            else:
                scaled_duration = 0.3 # Default if no words or zero total duration

            # Add natural pauses based on punctuation
            pause = 0.05
            if word.endswith((',', ';')):
                pause = 0.2
            elif word.endswith(('.', '!', '?')):
                pause = 0.4
            
            # Reduce pause for last word
            if i == len(words) - 1:
                pause = 0.1 # Shorter pause at the very end

            timings.append({
                'word': word,
                'start': current_time,
                'end': current_time + scaled_duration,
                'duration': scaled_duration,
                'confidence': 0.3  # Low confidence for estimation
            })

            current_time += scaled_duration + pause
        
        # Adjust final timing to ensure it doesn't exceed audio duration
        if timings and timings[-1]['end'] > audio_duration:
            timings[-1]['end'] = audio_duration

        print(f"Enhanced estimation complete: {len(timings)} word timings")
        return timings

    def create_subtitle_clip(self, word_timings, video_size, total_duration):
        """
        Create animated subtitle overlay clip using a solid magenta background for chroma keying.
        The text will be white with a black outline.
        """
        width, height = video_size

        # YouTube Shorts style settings
        font_size = min(width // 10, 80)  # Larger font for single word display

        # Cache font for better performance
        cached_font = self.get_best_font(font_size)

        def make_frame(t):
            # Create an RGB image with a solid magenta background.
            # This magenta will later be made transparent using mask_color.
            img = Image.new('RGB', (width, height), (255, 0, 255)) # Solid Magenta background
            draw = ImageDraw.Draw(img)

            font = cached_font

            current_word = None
            
            for timing in word_timings:
                if timing['start'] <= t < timing['end']:
                    current_word = timing['word']
                    break

            # If no word is currently being spoken, return a solid magenta frame
            if not current_word:
                return np.full((height, width, 3), (255, 0, 255), dtype=np.uint8) # Magenta frame

            # Obfuscate the word before drawing if applicable
            display_word = self._obfuscate_word(current_word)

            bbox = draw.textbbox((0, 0), display_word, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[0]

            text_x = (width - text_width) // 2
            text_y = (height - text_height) // 2

            # Draw black outline
            outline_width = 4 # Increased from 3 to 4 for a slightly more intense outline
            for adj_x in range(-outline_width, outline_width + 1):
                for adj_y in range(-outline_width, outline_width + 1):
                    if adj_x != 0 or adj_y != 0:
                        draw.text((text_x + adj_x, text_y + adj_y), display_word,
                                font=font, fill=(0, 0, 0)) # Solid black for outline

            # Draw main text in white
            draw.text((text_x, text_y), display_word, font=font, fill=(255, 255, 255)) # Solid white for text

            # Return the NumPy array of the RGB image
            return np.array(img)

        # Create the subtitle clip (RGB content with magenta background)
        subtitle_clip = VideoClip(make_frame, duration=total_duration)

        # Apply mask_color effect to make the magenta background transparent
        # This will convert the RGB clip to an RGBA clip with the magenta areas transparent.
        subtitle_clip = subtitle_clip.fx(fx.mask_color, color=(255, 0, 255), thr=1, s=1)
        # thr (threshold): 1 means exact match for color.
        # s (smoothness): 1 means sharp cut-off.

        return subtitle_clip


    def get_best_font(self, font_size):
        """
        Get the best available system font, prioritizing user-defined paths.
        It first checks the font paths provided during initialization,
        then common system paths for specific fonts, and finally generic Arial.
        """
        font_candidates = []

        # Prioritize user-defined fonts from arguments
        if self.font_path_main and os.path.exists(self.font_path_main):
            font_candidates.append(self.font_path_main)
        if self.font_path_fallback and os.path.exists(self.font_path_fallback):
            font_candidates.append(self.font_path_fallback)

        # Common system paths for Montserrat and Arial (cross-platform examples)
        # Windows
        font_candidates.append('C:/Windows/Fonts/Montserrat-SemiBold.ttf')
        font_candidates.append('C:/Windows/Fonts/arialbd.ttf')
        font_candidates.append('C:/Windows/Fonts/arial.ttf')
        # macOS
        font_candidates.append('/Library/Fonts/Montserrat-SemiBold.ttf')
        font_candidates.append('/System/Library/Fonts/Supplemental/Arial Bold.ttf')
        font_candidates.append('/System/Library/Fonts/Arial Bold.ttf')
        font_candidates.append('/System/Library/Fonts/Arial.ttf')
        # Linux
        font_candidates.append('/usr/share/fonts/truetype/montserrat/Montserrat-SemiBold.ttf')
        font_candidates.append('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf')
        font_candidates.append('/usr/share/fonts/TTF/arial.ttf')
        font_candidates.append('/usr/local/share/fonts/Arial.ttf')


        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except Exception as e:
                    print(f"Warning: Could not load font '{font_path}': {e}")
                    continue
        
        print("Warning: No suitable system font found. Using default Pillow font (may not display correctly).")
        return ImageFont.load_default()

    def create_intro_title_card(self, video_size, story_title, post_author, rewards_img_path, reddit_avatars_folder, duration=3):
        """
        Creates a static introductory title card clip with story title, rewards.png, and a random Reddit avatar
        overlaid on a white rectangle. The generated image for the title card will have a transparent
        background using a magenta chroma key.
        """
        width, height = video_size
        
        # Fixed card dimensions (percentages of video size)
        card_width = int(width * 0.9)
        card_height = int(height * 0.4) # Fixed height for the card
        
        # Internal padding for text within the card
        card_horizontal_padding = int(card_width * 0.05)
        card_vertical_padding = int(card_height * 0.05)

        # Max width for text within the card
        card_text_max_width = card_width - (2 * card_horizontal_padding)

        # Load and resize rewards.png
        rewards_img = None
        rewards_display_height = 0 # To track vertical space occupied by image
        
        try:
            if rewards_img_path and os.path.exists(rewards_img_path):
                rewards_img_raw = Image.open(rewards_img_path).convert("RGBA") # Ensure it has an alpha channel
                rewards_target_height = int(card_height * 0.08) # Roughly 8% of card height for icon
                rewards_img_width = int(rewards_img_raw.width * (rewards_target_height / rewards_img_raw.height))
                rewards_img = rewards_img_raw.resize((rewards_img_width, rewards_target_height), Image.Resampling.LANCZOS)
                rewards_display_height = rewards_img.height
                print(f"✅ Loaded and resized rewards.png to {rewards_img.width}x{rewards_img.height}")
            else:
                print(f"⚠️ Warning: rewards.png not found at {rewards_img_path}. Skipping image overlay.")
        except Exception as e:
            print(f"❌ Error loading or processing rewards.png: {e}. Skipping image overlay.")
            rewards_img = None

        # Load a random Reddit avatar
        selected_avatar_img = None
        avatar_size = int(card_height * 0.15) # Example size for the avatar
        try:
            if os.path.exists(reddit_avatars_folder) and os.path.isdir(reddit_avatars_folder):
                avatar_files = [f for f in os.listdir(reddit_avatars_folder) if f.lower().endswith('.png')]
                if avatar_files:
                    random_avatar_file = random.choice(avatar_files)
                    avatar_path = os.path.join(reddit_avatars_folder, random_avatar_file)
                    avatar_raw = Image.open(avatar_path).convert("RGBA")

                    # Resize to square
                    avatar_resized = avatar_raw.resize((avatar_size, avatar_size), Image.Resampling.LANCZOS)

                    # Create circular mask
                    mask = Image.new('L', (avatar_size, avatar_size), 0)
                    draw_mask = ImageDraw.Draw(mask)
                    draw_mask.ellipse((0, 0, avatar_size, avatar_size), fill=255)
                    
                    # Apply mask to avatar
                    selected_avatar_img = Image.new('RGBA', (avatar_size, avatar_size), (0, 0, 0, 0))
                    selected_avatar_img.paste(avatar_resized, (0, 0), mask)
                    print(f"✅ Loaded and processed random avatar: {random_avatar_file}")
                else:
                    print(f"⚠️ No PNG files found in '{reddit_avatars_folder}'. Skipping avatar overlay.")
            else:
                print(f"❌ Reddit avatars folder '{reddit_avatars_folder}' not found or not a directory. Skipping avatar overlay.")
        except Exception as e:
            print(f"❌ Error loading or processing Reddit avatar: {e}. Skipping avatar overlay.")
            selected_avatar_img = None


        # Determine font size for header text (smaller than main intro text)
        header_font_size = int(card_height * 0.06)
        header_font = self.get_best_font(header_font_size)

        # Calculate heights for positioning
        # Rewards (now above username)
        rewards_area_height = rewards_display_height + int(card_height * 0.01) if rewards_img else 0 # Small padding below rewards

        # Username line
        username_line_height = header_font.getbbox("Tg")[3] - header_font.getbbox("Tg")[1] # Height of one line of username text
        
        # Total header height occupied by avatar, rewards, username, and value
        header_elements_total_height = 0
        if selected_avatar_img:
            header_elements_total_height = max(header_elements_total_height, avatar_size)
        
        # Max of rewards_area_height and username_line_height for the text block (right of avatar)
        text_block_height = rewards_area_height + username_line_height
        header_elements_total_height = max(header_elements_total_height, text_block_height)

        header_elements_total_height += int(card_height * 0.03) # Additional padding for the entire header block


        # Define the target vertical space for main intro text (65% of remaining card_height after header)
        available_vertical_space_for_text = card_height - header_elements_total_height - (2 * card_vertical_padding)
        text_area_height = available_vertical_space_for_text * 0.65

        if text_area_height <= 0: # Fallback for minimal text area
            text_area_height = int(card_height * 0.1)


        # Function to wrap text based on estimated width to fit the card
        def wrap_text_for_size(text, font, max_width):
            lines = []
            words = text.split(' ')
            current_line = []
            
            dummy_img = Image.new('RGB', (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_img)

            for word in words:
                test_line = ' '.join(current_line + [word])
                try:
                    bbox = dummy_draw.textbbox((0, 0), test_line, font=font)
                    test_width = bbox[2] - bbox[0]
                except ValueError:
                    test_width = dummy_draw.textlength(test_line, font=font)

                if test_width <= max_width:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            lines.append(' '.join(current_line))
            return lines

        # Iteratively find the largest font size for the main intro text (story_title) that fits
        optimal_story_title_font_size = min(width // 20, 45) # Start with a reasonable max font size
        max_attempts = 30 
        current_attempt = 0

        best_wrapped_lines = []
        best_story_title_font = None

        while current_attempt < max_attempts and optimal_story_title_font_size > 5:
            current_font_for_test = self.get_best_font(optimal_story_title_font_size)
            if not current_font_for_test:
                break

            test_wrapped_lines = wrap_text_for_size(story_title, current_font_for_test, card_text_max_width)
            
            if test_wrapped_lines:
                try:
                    line_height_check_bbox = current_font_for_test.getbbox("Tg")
                    line_height_needed = (line_height_check_bbox[3] - line_height_check_bbox[1]) * len(test_wrapped_lines)
                except ValueError:
                    line_height_needed = current_font_for_test.getlength("Tg") * len(test_wrapped_lines)
            else:
                line_height_needed = 0

            if line_height_needed <= text_area_height:
                best_wrapped_lines = test_wrapped_lines
                best_story_title_font = current_font_for_test
                break
            
            optimal_story_title_font_size -= 1
            current_attempt += 1

        if not best_story_title_font:
            optimal_story_title_font_size = 10
            best_story_title_font = self.get_best_font(optimal_story_title_font_size)
            best_wrapped_lines = wrap_text_for_size(story_title, best_story_title_font, card_text_max_width)
            print("⚠️ Could not find optimal story title font size, using a small default.")


        # Now, make the frame with the determined font and wrapped text
        def make_card_frame(t):
            # Define radius here so it's in scope for this function
            radius = 20 # Adjust radius as needed

            # Create an RGB image with a solid magenta background for chroma keying
            full_frame_rgb = Image.new('RGB', (width, height), (255, 0, 255)) # Solid Magenta background
            full_frame_draw = ImageDraw.Draw(full_frame_rgb)
            
            # Create the card content on a transparent RGBA image first
            card_content_img = Image.new('RGBA', (card_width, card_height), (0, 0, 0, 0)) # Start with transparent
            card_content_draw = ImageDraw.Draw(card_content_img)
            
            # Draw the white background rectangle with rounded corners onto the transparent image
            card_content_draw.rounded_rectangle((0, 0, card_width, card_height), radius=radius, fill=(255, 255, 255, 255))

            # --- Positioning for elements in the top row ---
            
            # 1. Place Avatar (left-aligned, top-aligned in header block)
            avatar_x_on_card = card_horizontal_padding
            avatar_y_on_card = card_vertical_padding # Top of the entire header block

            if selected_avatar_img:
                card_content_img.paste(selected_avatar_img, (avatar_x_on_card, avatar_y_on_card), selected_avatar_img)
            
            # Calculate starting X for text block (rewards and username)
            text_block_start_x = card_horizontal_padding
            if selected_avatar_img:
                text_block_start_x += selected_avatar_img.width + int(card_horizontal_padding * 0.5)
            
            # 2. Place Rewards (above username, to the right of avatar)
            rewards_x_on_card = text_block_start_x
            rewards_y_on_card = card_vertical_padding # Aligned with top of avatar/header block
            
            if rewards_img:
                card_content_img.paste(rewards_img, (rewards_x_on_card, rewards_y_on_card), rewards_img)


            # 3. Draw Post Author (below rewards, to the right of avatar)
            post_author_x = text_block_start_x
            post_author_y = card_vertical_padding + rewards_area_height # Start username below rewards, using rewards_area_height for vertical space

            card_content_draw.text((post_author_x, post_author_y), post_author, font=header_font, fill=(0, 0, 0))


            # Calculate total text block height for centering main text
            if best_wrapped_lines:
                main_text_line_height = (best_story_title_font.getbbox("Tg")[3] - best_story_title_font.getbbox("Tg")[1])
                main_text_block_height = main_text_line_height * len(best_wrapped_lines)
            else:
                main_text_block_height = 0

            # Calculate the starting Y position for the main text to center it within its allocated area
            # This area starts after the header_elements_total_height and card_vertical_padding
            main_text_area_start_y = header_elements_total_height + card_vertical_padding
            
            current_y_on_card = main_text_area_start_y + (text_area_height - main_text_block_height) // 2
            
            # Ensure text doesn't start too high (should be below the entire header block)
            current_y_on_card = max(current_y_on_card, header_elements_total_height + int(card_height * 0.02))

            # Draw each wrapped line of main intro text (story_title)
            for line in best_wrapped_lines:
                # Text is left-aligned within its text area
                line_x_on_card = card_horizontal_padding

                # Draw black outline for main text
                outline_width_card_main_text = 2 
                for adj_x in range(-outline_width_card_main_text, outline_width_card_main_text + 1):
                    for adj_y in range(-outline_width_card_main_text, outline_width_card_main_text + 1):
                        if adj_x != 0 or adj_y != 0:
                            card_content_draw.text((line_x_on_card + adj_x, current_y_on_card + adj_y), line,
                                           font=best_story_title_font, fill=(0, 0, 0))

                # Draw main text on card in black
                card_content_draw.text((line_x_on_card, current_y_on_card), line, font=best_story_title_font, fill=(0, 0, 0))
                current_y_on_card += main_text_line_height # Move to next line

            # Position the card content onto the full_frame_rgb
            card_x_pos = (width - card_width) // 2
            card_y_pos = (height - card_height) // 2
            full_frame_rgb.paste(card_content_img, (card_x_pos, card_y_pos), card_content_img) # Use card_content_img as mask

            return np.array(full_frame_rgb)

        # The intro clip is created with a magenta background
        intro_clip = VideoClip(make_card_frame, duration=duration).set_fps(24)
        # Then, apply mask_color to make the magenta transparent
        intro_clip = intro_clip.fx(fx.mask_color, color=(255, 0, 255), thr=1, s=1)

        return intro_clip

    def _synthesize_and_append(self, text_to_synthesize, voice_id, all_audio_segments_list):
        """Helper to synthesize a chunk and append to the list."""
        print(f"Synthesizing chunk (length {len(text_to_synthesize)}): '{text_to_synthesize[:50]}...'")
        try:
            response = self.polly_client.synthesize_speech(
                VoiceId=voice_id,
                OutputFormat='mp3',
                Text=text_to_synthesize,
                Engine='neural'
            )
            # Use a temporary file for each chunk with a random component for uniqueness
            temp_audio_file = os.path.join(self.temp_dir, f"polly_chunk_{len(all_audio_segments_list)}_{random.randint(0,1000)}.mp3")
            with open(temp_audio_file, 'wb') as file:
                file.write(response['AudioStream'].read())
            all_audio_segments_list.append(AudioSegment.from_mp3(temp_audio_file))
        except Exception as e:
            print(f"❌ Error synthesizing chunk: {e}")
            raise # Re-raise to stop processing if a chunk fails


    def generate_tts_audio(self, text, output_path, voice_gender='J'):
        """
        Generate TTS audio using Amazon Polly. Handles text chunking for long inputs.
        Requires AWS credentials to be configured.
        voice_gender: 'J' for Joanna (female), 'M' for Matthew (male). Defaults to Joanna.
        """
        if not self.polly_available or self.polly_client is None:
            print("❌ Amazon Polly client not initialized. Skipping TTS generation.")
            return False

        print(f"Generating TTS audio using Amazon Polly for {len(text)} characters...")

        # Determine VoiceId based on gender preference
        voice_id = 'Matthew' if voice_gender.upper() == 'M' else 'Joanna'
        print(f"Selected voice: {voice_id}")

        all_audio_segments = []
        
        # Split by sentences. Use a slightly more robust regex if needed for varied punctuation.
        sentences = re.split(r'(?<=[.!?])\s+|\n', text)
        sentences = [s.strip() for s in sentences if s.strip()] # Clean up empty strings

        current_chunk_text = ""

        for sentence in sentences:
            if len(sentence) > self.POLLY_MAX_CHARS:
                # If a single sentence is too long, break it into smaller fixed-size pieces
                if current_chunk_text: # Synthesize any preceding accumulated chunk
                    self._synthesize_and_append(current_chunk_text, voice_id, all_audio_segments)
                    current_chunk_text = ""

                print(f"⚠️ Sentence too long ({len(sentence)} chars), splitting into sub-chunks.")
                # Break long sentence into smaller chunks
                for i in range(0, len(sentence), self.POLLY_MAX_CHARS):
                    sub_chunk = sentence[i:i + self.POLLY_MAX_CHARS]
                    self._synthesize_and_append(sub_chunk, voice_id, all_audio_segments)
            else:
                # Add sentence to current chunk if it fits
                if len(current_chunk_text) + len(sentence) + (1 if current_chunk_text else 0) > self.POLLY_MAX_CHARS:
                    self._synthesize_and_append(current_chunk_text, voice_id, all_audio_segments)
                    current_chunk_text = sentence
                else:
                    current_chunk_text += (" " if current_chunk_text else "") + sentence
        
        # Synthesize any remaining chunk
        if current_chunk_text:
            self._synthesize_and_append(current_chunk_text, voice_id, all_audio_segments)

        if not all_audio_segments:
            print("No audio segments were generated.")
            return False

        # Concatenate all audio segments
        combined_audio = AudioSegment.empty()
        for audio_segment in all_audio_segments:
            combined_audio += audio_segment

        # Export the combined audio to the final output path
        try:
            combined_audio.export(output_path, format="mp3")
            print(f"Amazon Polly TTS audio (combined from chunks) saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error exporting combined TTS audio: {e}")
            return False

    def create_subtitle_video(self, text, background_video_path, output_path, story_title_arg, post_author_arg, rewards_img_path, reddit_avatars_folder, voice_gender_arg='J'):
        """Create video with animated subtitles using best available timing method"""
        print("🎬 Processing subtitle video with enhanced accuracy...")

        # Define a transition buffer duration (e.g., 0.5 seconds)
        transition_buffer_duration = 0.5 # Seconds of blank background between intro and main content

        # Generate TTS audio for main content
        main_audio_path = os.path.join(self.temp_dir, "main_tts_audio.mp3")
        if not self.generate_tts_audio(text, main_audio_path, voice_gender=voice_gender_arg):
            print("Exiting due to main content TTS audio generation failure.")
            return False

        # Load main content TTS audio and get duration
        try:
            main_tts_audio = AudioFileClip(main_audio_path)
            main_audio_duration = main_tts_audio.duration
            print(f"Main audio duration: {main_audio_duration:.2f}s")
        except Exception as e:
            print(f"Error loading main TTS audio: {e}")
            return False

        # Define intro clip duration and text
        story_title_content = story_title_arg # Already handled by main() parsing story.json
        post_author_content = post_author_arg # Already handled by main() parsing story.json
        
        # Generate TTS audio for intro text
        intro_audio_path = os.path.join(self.temp_dir, "intro_tts_audio.mp3")
        # For intro audio, use the story title as the spoken text
        if not self.generate_tts_audio(story_title_content, intro_audio_path, voice_gender=voice_gender_arg):
            print("Exiting due to intro TTS audio generation failure.")
            intro_audio_duration = 1.0 # Default short silent intro
        else:
            try:
                intro_tts_audio = AudioFileClip(intro_audio_path)
                intro_audio_duration = intro_tts_audio.duration
                print(f"Intro audio duration: {intro_audio_duration:.2f}s")
            except Exception as e:
                print(f"Error loading intro TTS audio, using default duration: {e}")
                intro_audio_duration = 1.0 # Fallback

        # Calculate total final video duration (intro + transition buffer + main content)
        total_final_video_duration = intro_audio_duration + transition_buffer_duration + main_audio_duration

        # Load background video
        background_full = None # Initialize to None for scope
        final_video = None
        subtitle_clip = None
        intro_title_card_clip = None # Initialize intro clip

        # Define target_video_width and target_video_height at the beginning
        target_video_width = 1080 # Standard width for vertical video (e.g., YouTube Shorts)
        target_video_height = 1920 # Standard height for vertical video

        try:
            background_full = VideoFileClip(background_video_path)

            print(f"Original background video size: {background_full.w}x{background_full.h}")
            print(f"Original background video duration: {background_full.duration:.2f}s")

            # Check if background video is long enough for the ENTIRE composite video
            if background_full.duration < total_final_video_duration:
                print(f"❌ Error: Background video ({background_full.duration:.2f}s) is shorter than the total required video duration ({total_final_video_duration:.2f}s).")
                print("Please provide a background video that is longer than or equal to the total video duration to avoid looping.")
                background_full.close() # Close the clip before exiting
                return False

            # Calculate the maximum possible start time for the background video segment
            max_start_time = background_full.duration - total_final_video_duration
            
            # Choose a random start time within the valid range
            random_start_time = random.uniform(0, max_start_time)
            print(f"Random background video start time chosen: {random_start_time:.2f}s for total duration of {total_final_video_duration:.2f}s.")

            # Subclip the background video for the entire required duration
            background_for_composite = background_full.subclip(random_start_time, random_start_time + total_final_video_duration)

            # Calculate scaling factors for the background video segment
            scale_w = target_video_width / background_for_composite.w
            scale_h = target_video_height / background_for_composite.h

            # Choose the larger scale factor to ensure the video covers the target dimensions
            scale_factor = max(scale_w, scale_h)

            # Resize the background video segment to fill the target dimensions, potentially exceeding
            background_for_composite = background_for_composite.resize(scale_factor)
            print(f"Background video resized to (after scaling): {background_for_composite.w}x{background_for_composite.h}")

            # Crop the resized video to the exact target dimensions, centering the crop
            final_background_video_clip = background_for_composite.crop(
                x_center=background_for_composite.w / 2,
                y_center=background_for_composite.h / 2,
                width=target_video_width,
                height=target_video_height
            )
            print(f"Background video cropped to final size: {final_background_video_clip.w}x{final_background_video_clip.h}")
            print(f"Final background video clip duration: {final_background_video_clip.duration:.2f}s")

            video_size = (final_background_video_clip.w, final_background_video_clip.h)
            print(f"Final video size for composition: {video_size}")

        except Exception as e:
            print(f"Error loading or adjusting background video: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return False

        # The background clip for the intro and buffer duration
        background_for_intro_clip = final_background_video_clip.subclip(0, intro_audio_duration + transition_buffer_duration)

        # Create the intro title card clip
        print(f"Creating intro title card for {intro_audio_duration:.2f}s with title: '{story_title_content}' by '{post_author_content}'")
        # Pass rewards_img_path and reddit_avatars_folder to create_intro_title_card
        intro_title_card_clip = self.create_intro_title_card(video_size, story_title_content, post_author_content, rewards_img_path, reddit_avatars_folder, duration=intro_audio_duration)
        intro_title_card_clip = intro_title_card_clip.set_opacity(1.0) # Ensure it's fully opaque


        # Clean text and get the most accurate word timings possible for main content
        clean_text = self.clean_text(text)
        word_timings = self.analyze_speech_timing(main_audio_path, clean_text)

        # Print timing method used
        if word_timings and len(word_timings) > 0:
            avg_confidence = sum(t.get('confidence', 1.0) for t in word_timings) / len(word_timings)
            if avg_confidence > 0.8:
                timing_method = "Whisper AI (High Accuracy)"
            else:
                timing_method = "Estimation (Basic Accuracy)"

            print(f"🎯 Timing Method: {timing_method}")
            print(f"📊 Average Confidence: {avg_confidence:.2f}")

        print(f"Creating subtitle overlay for main content with {len(word_timings)} words...")

        # Create main subtitle clip (duration matches main audio duration)
        subtitle_clip = self.create_subtitle_clip(word_timings, video_size, main_audio_duration)
        # No set_start here, as it will be set relative to its parent composite later

        # Create a silent audio clip for the transition buffer
        silent_buffer_audio_segment = AudioSegment.silent(duration=int(transition_buffer_duration * 1000))
        silent_buffer_audio_path = os.path.join(self.temp_dir, "silent_buffer_audio.mp3")
        silent_buffer_audio_segment.export(silent_buffer_audio_path, format="mp3")
        silent_buffer_audio_clip = AudioFileClip(silent_buffer_audio_path)


        # Concatenate intro TTS audio, silent buffer audio, and main TTS audio
        # Define main_content_start_time correctly
        main_content_start_time = intro_audio_duration + transition_buffer_duration 
        if 'intro_tts_audio' in locals() and intro_tts_audio is not None:
             final_audio_track = CompositeAudioClip([
                intro_tts_audio.set_start(0),
                silent_buffer_audio_clip.set_start(intro_audio_duration), # Buffer starts after intro audio
                main_tts_audio.set_start(main_content_start_time) # Main audio starts after buffer
            ])
        else: # Fallback to silent intro audio if TTS failed for intro
            silent_intro_audio_segment = AudioSegment.silent(duration=int(intro_audio_duration * 1000))
            silent_intro_audio_path = os.path.join(self.temp_dir, "silent_intro_audio_fallback.mp3")
            silent_intro_audio_segment.export(silent_intro_audio_path, format="mp3")
            silent_intro_audio_clip = AudioFileClip(silent_intro_audio_path)
            final_audio_track = CompositeAudioClip([
                silent_intro_audio_clip.set_start(0),
                silent_buffer_audio_clip.set_start(intro_audio_duration),
                main_tts_audio.set_start(main_content_start_time)
            ])


        # Composite everything
        print("🎬 Compositing final video...")
        try:
            # The intro composite (background video for intro duration + title card)
            intro_composite = CompositeVideoClip([
                final_background_video_clip.subclip(0, intro_audio_duration + transition_buffer_duration), # Background for intro + buffer
                intro_title_card_clip.set_start(0) # Title card at start of this composite
            ], size=video_size).set_duration(intro_audio_duration + transition_buffer_duration) # Total duration for this block


            # The main content composite (unblurred background + subtitles)
            # The subtitle clip is placed at 0 relative to the start of this composite,
            # and then this composite is itself positioned at `main_content_start_time`
            main_content_clip = CompositeVideoClip([
                final_background_video_clip.subclip(main_content_start_time, total_final_video_duration), # Background for main content
                subtitle_clip.set_start(0) # Subtitles should start at 0 relative to main_content_clip's start
            ], size=video_size).set_duration(main_audio_duration) # This composite's own duration


            final_video = CompositeVideoClip([
                intro_composite, # This now includes the buffer as part of its duration
                main_content_clip.set_start(intro_composite.duration) # Explicitly set start time for main content to be right after intro_composite
            ], size=video_size).set_audio(final_audio_track)

            # Ensure final video duration matches total composite duration precisely
            final_video = final_video.set_duration(total_final_video_duration)

            # Write final video
            print(f"🎬 Rendering final video: {output_path}")
            final_video.write_videofile(
                output_path,
                fps=final_background_video_clip.fps, # Match background FPS for consistency
                codec='libx264', # Reverted to libx264 for broader compatibility
                audio_codec='aac',
                temp_audiofile=os.path.join(self.temp_dir, 'temp-audio.m4a'),
                remove_temp=True,
                verbose=True, # Set to True to display progress bar
                logger='bar', # Use 'bar' to display a simple ASCII progress bar
                preset="fast" # Changed to "fast" for quicker rendering
            )

            print(f"✅ Enhanced subtitle video created successfully: {output_path}")
            success = True

        except PermissionError as e: # Catch PermissionError specifically
            print(f"❌ Error: Permission denied when writing output file. Please check:")
            print(f"   1. Do you have write permissions to the directory '{os.path.dirname(output_path)}'?")
            print(f"   2. Is the output file '{output_path}' currently open in another program (e.g., a video player)? If so, close it and try again.")
            print(f"   3. Try saving to a different location, like your Desktop or a temporary folder, to rule out directory-specific issues.")
            import traceback
            traceback.print_exc()
            success = False
        except Exception as e:
            print(f"❌ An unexpected error occurred during compositing: {e}")
            import traceback
            traceback.print_exc()
            success = False

        finally:
            # Cleanup clips (MoviePy resources)
            # Ensure all clips are closed to release resources and file locks
            if 'background_full' in locals() and background_full is not None:
                background_full.close() # Close the original full clip
            if 'background_for_composite' in locals() and background_for_composite is not None:
                background_for_composite.close()
            if 'final_background_video_clip' in locals() and final_background_video_clip is not None:
                final_background_video_clip.close()
            if 'main_tts_audio' in locals() and main_tts_audio is not None:
                main_tts_audio.close()
            if 'intro_tts_audio' in locals() and intro_tts_audio is not None:
                intro_tts_audio.close()
            if 'subtitle_clip' in locals() and subtitle_clip is not None:
                subtitle_clip.close()
            if 'intro_title_card_clip' in locals() and intro_title_card_clip is not None:
                intro_title_card_clip.close()
            if 'silent_buffer_audio_clip' in locals() and silent_buffer_audio_clip is not None:
                silent_buffer_audio_clip.close()
            if 'final_audio_track' in locals() and final_audio_track is not None:
                final_audio_track.close()
            if 'intro_composite' in locals() and intro_composite is not None:
                intro_composite.close()
            if 'main_content_clip' in locals() and main_content_clip is not None:
                main_content_clip.close()
            if 'final_video' in locals() and final_video is not None:
                final_video.close()
            
            # Call the class's cleanup method for the temporary directory
            self.cleanup() # This will handle the temp directory cleanup

        return success

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory {self.temp_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate YouTube Shorts-style subtitled videos with maximum accuracy')
    
    parser.add_argument('background_video', type=str,
                        help='Path to the background video file.')
    parser.add_argument('output', type=str,
                        help='Desired path for the output video file.')
    
    parser.add_argument('--story-json', type=str, 
                        default='story.json', # Default to 'story.json' in current directory
                        help='Path to a JSON file containing "story_title", "post_author", and "story_text". Defaults to "story.json".')
    parser.add_argument('--obfuscation-json', type=str, 
                        default='obfuscation.json', # Default to 'obfuscation.json' in current directory
                        help='Path to a JSON file containing words to obfuscate and their alternatives. Defaults to "obfuscation.json".')
    parser.add_argument('--rewards-img', type=str,
                        default='rewards.png', # Default to 'rewards.png' in current directory
                        help='Path to the rewards.png image file for the intro card. Defaults to "rewards.png".')
    parser.add_argument('--reddit-avatars-folder', type=str,
                        default='RedditAvatars', # Default to 'RedditAvatars' subfolder
                        help='Path to the folder containing Reddit avatar PNGs. Defaults to "RedditAvatars".')
    parser.add_argument('--font-main-path', type=str,
                        default='Montserrat-SemiBold.ttf', # Default to a common font name in current directory
                        help='Path to the main font file (e.g., Montserrat-SemiBold.ttf). Script will search common system paths if not found.')
    parser.add_argument('--font-fallback-path', type=str,
                        default='Arial.ttf', # Default to Arial.ttf in current directory
                        help='Path to a fallback font file (e.g., Arial.ttf). Script will search common system paths if not found.')


    parser.add_argument('--force-estimation', action='store_true',
                        help='Skip advanced analysis and use estimation only (faster but less accurate)')
    parser.add_argument('--voice-gender', type=str, default='J', choices=['J', 'M'],
                        help="Choose voice gender for Amazon Polly: 'J' for Joanna (female), 'M' for Matthew (male). Defaults to 'J'.")


    args = parser.parse_args()

    story_text = ""
    story_title = None
    post_author = None

    # Load story data from the specified JSON file
    try:
        if os.path.exists(args.story_json):
            with open(args.story_json, 'r', encoding='utf-8') as f:
                story_data = json.load(f)
            story_text = story_data.get('story_text', '')
            story_title = story_data.get('story_title', 'A New Story') # Provide default if not present in JSON
            post_author = story_data.get('post_author', '@RedditStories') # Provide default if not present in JSON
            print(f"Loaded story from JSON: Title='{story_title}', Author='{post_author}' from: {args.story_json}")
        else:
            print(f"Error: JSON story file not found at: {args.story_json}")
            return 1 # Exit if file not found
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in story file: {args.story_json}")
        return 1 # Exit if JSON is invalid
    except Exception as e:
        print(f"Error reading story JSON: {e}")
        return 1 # Exit for other file reading errors


    if not story_text:
        print("Error: 'story_text' field is missing or empty in the provided JSON file.")
        return 1

    # Validate background video file
    if not os.path.exists(args.background_video):
        print(f"Error: Background video not found: {args.background_video}")
        return 1

    # Show available analysis methods
    print("🔍 Available timing analysis methods:")
    if WHISPER_AVAILABLE:
        print("   ✅ Whisper AI (Highest Accuracy)")
    else:
        print("   ❌ Whisper AI (Install: pip install openai-whisper torch)")

    print("   ✅ Enhanced Estimation (Basic Accuracy)")
    print()

    # Create subtitle generator, passing file paths for fonts and avatars
    generator = RedditTTSSubtitles(
        font_main_path=args.font_main_path,
        font_fallback_path=args.font_fallback_path,
        reddit_avatars_folder=args.reddit_avatars_folder,
        obfuscation_file_path=args.obfuscation_json
    )

    try:
        # Pass all necessary args to create_subtitle_video
        success = generator.create_subtitle_video(
            text=story_text,
            background_video_path=args.background_video,
            output_path=args.output,
            story_title_arg=story_title,
            post_author_arg=post_author,
            rewards_img_path=args.rewards_img, # Pass rewards image path
            reddit_avatars_folder=args.reddit_avatars_folder, # Pass avatars folder path
            voice_gender_arg=args.voice_gender
        )
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        pass # Cleanup is handled internally by the generator instance

if __name__ == "__main__":
    sys.exit(main())
