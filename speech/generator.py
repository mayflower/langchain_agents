from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_google_vertexai import ChatVertexAI
import queue
import threading
import random
import os
import re
import google.cloud.texttospeech as tts
from pydub import AudioSegment
from pydub.playback import play


class GenerateAndPlayBack:
    """Generates an LLM answer via Gemini-Pro 1.5 and plays it back via Google TTS."""

    def __init__(self, model=ChatVertexAI(model="gemini-1.5-pro-preview-0409")):
        self.model = model
        self.sentence_queue = queue.Queue()
        self.wav_queue = queue.Queue()
        self.sentences = []

    """Converts text to speech using Google TTS and saves it to a .wav file."""

    def text_to_wav(self, voice_name: str, text: str):
        language_code = "de-DE"
        text_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(
            language_code=language_code, name="de-DE-Standard-C"
        )
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

        client = tts.TextToSpeechClient()
        response = client.synthesize_speech(
            input=text_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        filename = f"{voice_name}.wav"
        with open(filename, "wb") as out:
            out.write(response.audio_content)
        return filename

    """Plays a .wav file using PyDub."""

    def play_wav(self, filename: str):
        audio = AudioSegment.from_wav(filename)
        play(audio)

    """Processes the sentence queue and generates .wav files for each sentence."""

    def process_queue(self):
        while True:
            item = self.sentence_queue.get()
            if item is None:
                self.wav_queue.put("STOP_QUEUE_WAV")
                break

            filename = self.text_to_wav(str(random.randint(1, 1000)), item)
            self.wav_queue.put(filename)
            print("\n", "sentences", item, "\n")

    """Processes the .wav queue and plays back the .wav files."""

    def process_wav_queue(self):
        while True:
            item = self.wav_queue.get()
            print("item", item)
            if item is None:
                continue
            if item == "STOP_QUEUE_WAV":
                break
            print("playing", item)
            self.play_wav(item)
            os.remove(item)

    """Generates an LLM answer and puts each sentence in the sentence queue for processing
    
    Is configured for text generation with Gemini Pro. For use with other models the generate_and_play may need to be adjusted.
    This is because Gemini Pro streaming does not return single tokens but chunks of text. Sentences are actually split by . after a character.
    It will not work if single tokens are returned by different models."""

    def generate_and_play(self, result):
        """Start the processing queue threads."""
        t = threading.Thread(target=self.process_queue)
        tw = threading.Thread(target=self.process_wav_queue)
        t.start()
        tw.start()

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    "Du bist ein Sprachassistent. Formuliere deine Antworten immer in einer Art die einfach vorgelesen werden kann. Lasse vor allem Sonderzeichen wie Sternchen oder Emojis weg."
                ),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        actual_sentence = ""
        for chunk in (prompt | self.model).stream({"input": result}):
            """Collect tokens and split them into sentences."""
            match = re.search(r"[a-zA-Z]\.", chunk.content)
            if match:
                start, end = match.span()
                start_string = chunk.content[: start + 2]
                end_string = chunk.content[end:]
                actual_sentence += start_string
                self.sentence_queue.put(actual_sentence)
                actual_sentence = end_string
            else:
                actual_sentence += chunk.content
        """Put the last sentence in the queue and stop the queue by putting a None."""
        self.sentence_queue.put(actual_sentence)
        self.sentence_queue.put(None)

        """Wait for the threads to finish."""
        t.join()
        tw.join()
