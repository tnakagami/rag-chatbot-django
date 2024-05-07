# Generated by Django 4.2.11 on 2024-05-03 22:36

import chatbot.models.agents
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='embedding',
            name='emb',
            field=models.IntegerField(choices=[(1, 'Open AI'), (2, 'Azure'), (4, 'Amazon Bedrock'), (5, 'Fireworks (Mixtral)'), (6, 'Ollama'), (7, 'GEMINI')], default=1, validators=[chatbot.models.agents.AgentType.get_embedding_validator], verbose_name='Embedding type'),
        ),
    ]
