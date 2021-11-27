# Generated by Django 3.2 on 2021-10-14 08:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '__first__'),
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userplaylist',
            name='playlists',
        ),
        migrations.AddField(
            model_name='userplaylist',
            name='playlist',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.DO_NOTHING, to='api.playlist'),
        ),
    ]