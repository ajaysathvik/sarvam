"""
Sarvam AI Real-Time Voice Agent  (SDK edition)
================================================
Pipeline: Microphone → Saaras v3 (STT) → sarvam-m1 (LLM) → Bulbul v3 (TTS) → Speaker

Requirements:
    pip install sarvamai pyaudio python-dotenv

Usage:
    python voice_agent.py
"""

import io
import os
import struct
import wave

import pyaudio
from dotenv import load_dotenv
from sarvamai import SarvamAI
from sarvamai.play import play
from guardrails import check_input, check_output
from rag import retrieve_context

# ─── Config ───────────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.getenv("SARVAM_API_KEY", "")

client = SarvamAI(api_subscription_key=API_KEY)

# Audio recording
SAMPLE_RATE      = 16_000
CHANNELS         = 1
FORMAT           = pyaudio.paInt16
CHUNK_SIZE       = 1024
SILENCE_THRESHOLD = 600    # RMS; raise if mic picks up background noise
SILENCE_DURATION = 1.8     # seconds of silence → end of utterance
MAX_RECORD_SECS  = 30      # hard cap

# TTS
TTS_LANGUAGE = "en-IN"     # hi-IN | en-IN | ta-IN | te-IN | kn-IN | ml-IN | ...
TTS_SPEAKER  = "shubh"     # anushka | priya | shubh | rahul | kavya | ...

# LLM
SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant powered by Sarvam AI. "
    "Keep answers short and conversational — 1-3 sentences max. "
    "Reply in the same language the user speaks."
)

EXIT_KEYWORDS = {"quit", "exit", "bye", "goodbye", "stop"}

# ─── Helpers ──────────────────────────────────────────────────────────────────

ANSI = {
    "MIC": "\033[94m",   # blue
    "STT": "\033[93m",   # yellow
    "LLM": "\033[92m",   # green
    "TTS": "\033[96m",   # cyan
    "ERR": "\033[91m",   # red
    "SYS": "\033[95m",   # magenta
    "RST": "\033[0m",
}

def log(tag: str, msg: str):
    print(f"{ANSI[tag]}[{tag}]{ANSI['RST']} {msg}")


def rms(data: bytes) -> float:
    count = len(data) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"{count}h", data[: count * 2])
    return (sum(s * s for s in shorts) / count) ** 0.5


def frames_to_wav(frames: list[bytes], rate: int = SAMPLE_RATE) -> io.BytesIO:
    """Pack PCM frames into an in-memory WAV file object."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
    buf.seek(0)
    buf.name = "audio.wav"   # sarvamai SDK needs a .name attribute
    return buf


# ─── 1. Record from Microphone ────────────────────────────────────────────────

def record_utterance(pa: pyaudio.PyAudio) -> io.BytesIO:
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    log("MIC", "Listening… (speak now)")

    frames: list[bytes] = []
    speaking = False
    silent_chunks = 0
    silence_limit = int(SAMPLE_RATE / CHUNK_SIZE * SILENCE_DURATION)
    max_chunks    = int(SAMPLE_RATE / CHUNK_SIZE * MAX_RECORD_SECS)

    while len(frames) < max_chunks:
        data  = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        level = rms(data)
        frames.append(data)

        if level > SILENCE_THRESHOLD:
            speaking      = True
            silent_chunks = 0
        elif speaking:
            silent_chunks += 1
            if silent_chunks >= silence_limit:
                break   # enough silence after speech detected

    stream.stop_stream()
    stream.close()

    return frames_to_wav(frames)


# ─── 2. STT — Saaras v3 ───────────────────────────────────────────────────────

def transcribe(wav_buf: io.BytesIO) -> str:
    log("STT", "Transcribing…")
    try:
        response = client.speech_to_text.transcribe(
            file=wav_buf,
            model="saaras:v3",
            mode="transcribe",        # transcribe | translate | verbatim | translit | codemix
        )
        # SDK returns an object; grab the transcript string
        transcript = getattr(response, "transcript", None)
        text = transcript if transcript is not None else str(response)
        return text.strip()
    except Exception as e:
        log("ERR", f"STT failed: {e}")
        return ""


# ─── 3. LLM — sarvam-m1 ──────────────────────────────────────────────────────

def chat(user_text: str, history: list[dict], context: str = "") -> str:
    log("LLM", "Thinking…")
    system = SYSTEM_PROMPT
    if context:
        system += f"\n\nRelevant knowledge:\n{context}"
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions(
            messages=messages,
            temperature=0.7,
            top_p=1,
            max_tokens=300,
        )
        # SDK returns object with choices list
        if hasattr(response, "choices"):
            return response.choices[0].message.content.strip()
        return str(response).strip()
    except Exception as e:
        log("ERR", f"LLM failed: {e}")
        return "I'm sorry, I ran into a problem. Could you repeat that?"


# ─── 4. TTS — Bulbul v3 ───────────────────────────────────────────────────────

def speak(text: str) -> None:
    log("TTS", "Speaking…")
    try:
        response = client.text_to_speech.convert(
            text=text,
            target_language_code=TTS_LANGUAGE,
            speaker=TTS_SPEAKER,
            model="bulbul:v3",
        )
        play(response)
    except Exception as e:
        log("ERR", f"TTS failed: {e}")


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    log("SYS", "=" * 55)
    log("SYS", "  Sarvam AI Voice Agent")
    log("SYS", "  STT: Saaras v3  |  LLM: sarvam-m1  |  TTS: Bulbul v3")
    log("SYS", f"  Language: {TTS_LANGUAGE}  |  Speaker: {TTS_SPEAKER}")
    log("SYS", "  Say 'quit' or 'exit' to stop. Press Ctrl+C to force-quit.")
    log("SYS", "=" * 55)

    pa = pyaudio.PyAudio()
    history: list[dict] = []

    try:
        while True:
            # Step 1 — record
            wav_buf = record_utterance(pa)

            # Step 2 — transcribe
            user_text = transcribe(wav_buf)

            if not user_text:
                log("STT", "Nothing detected — try again.")
                continue

            log("STT", f'You: "{user_text}"')

            # Exit check
            if any(kw in user_text.lower() for kw in EXIT_KEYWORDS):
                farewell = "Goodbye! Have a great day!"
                log("LLM", f'Agent: "{farewell}"')
                speak(farewell)
                break

            # Guardrail — input check
            is_safe, reason = check_input(user_text)
            if not is_safe:
                refusal = "I'm sorry, I can't help with that topic."
                log("ERR", f"Input blocked: {reason}")
                speak(refusal)
                continue

            # RAG — retrieve relevant context from PDFs
            context = retrieve_context(user_text)
            if context:
                log("LLM", f"RAG: injecting {len(context.split())} words of context")

            # Step 3 — LLM
            reply = chat(user_text, history, context=context)

            # Guardrail — output check
            is_safe, reply = check_output(reply)
            if not is_safe:
                reply = "I'm sorry, I can't share that response."
                log("ERR", "Output blocked by guardrail.")

            log("LLM", f'Agent: "{reply}"')

            # Update conversation history (keep last 10 turns = 20 messages)
            history += [
                {"role": "user",      "content": user_text},
                {"role": "assistant", "content": reply},
            ]
            history = history[-20:]

            # Step 4 — TTS
            speak(reply)

    except KeyboardInterrupt:
        log("SYS", "\nStopped by user. Goodbye!")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
