from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, UploadedFile, ProcessingProgress


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ("email", "name", "username", "is_staff", "date_joined")
    list_filter = ("is_staff", "is_superuser", "is_active", "date_joined")
    search_fields = ("email", "name", "username")
    ordering = ("email",)

    fieldsets = UserAdmin.fieldsets + (("Additional Info", {"fields": ("name",)}),)
    add_fieldsets = UserAdmin.add_fieldsets + (
        ("Additional Info", {"fields": ("name", "email")}),
    )


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ("filename", "user", "uploaded_at", "processed")
    list_filter = ("processed", "uploaded_at")
    search_fields = ("filename", "user__email")
    readonly_fields = ("uploaded_at",)


@admin.register(ProcessingProgress)
class ProcessingProgressAdmin(admin.ModelAdmin):
    list_display = ("user", "progress", "updated_at")
    list_filter = ("updated_at",)
    search_fields = ("user__email",)
    readonly_fields = ("updated_at",)
