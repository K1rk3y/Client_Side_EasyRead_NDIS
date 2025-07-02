from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # Add extra fields here if needed, e.g. phone = models.CharField(max_length=20, blank=True)
    role = models.CharField(max_length=50, blank=True, default="user")
    profilepic = models.TextField(blank=True, null=True, help_text="Base64-encoded profile picture")
    AItoken = models.TextField(blank=True, null=True, help_text="OpenAI token for this user")
