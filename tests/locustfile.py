from locust import HttpUser, task, between
import os

AUDIO_FILE = "test_audio.wav"

class ASRUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def transcribe(self):
        with open(AUDIO_FILE, "rb") as f:
            self.client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")}
            )

    @task(3)
    def health_check(self):
        self.client.get("/health")
