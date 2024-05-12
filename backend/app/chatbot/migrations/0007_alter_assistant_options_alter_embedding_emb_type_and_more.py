# Generated by Django 4.2.13 on 2024-05-11 14:51

import chatbot.models.agents
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0006_alter_assistant_tools_alter_embedding_emb_type'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='assistant',
            options={'ordering': ['pk']},
        ),
        migrations.AlterField(
            model_name='embedding',
            name='emb_type',
            field=models.IntegerField(choices=[(1, 'Open AI'), (2, 'Azure'), (4, 'Amazon Bedrock'), (5, 'Fireworks (Mixtral)'), (6, 'Ollama'), (7, 'GEMINI')], default=1, validators=[chatbot.models.agents.AgentType.get_embedding_validator], verbose_name='Embedding type'),
        ),
        migrations.CreateModel(
            name='DocumentFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='255 characters or fewer.', max_length=255, verbose_name='Document name')),
                ('assistant', models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, related_name='docfiles', to='chatbot.assistant', verbose_name='Base assistant of document files')),
            ],
        ),
        migrations.AddField(
            model_name='thread',
            name='docfiles',
            field=models.ManyToManyField(blank=True, related_name='docfiles', to='chatbot.documentfile', verbose_name='Document files used in RAG'),
        ),
    ]
