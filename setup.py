from setuptools import setup, find_packages
import os

# Leer la descripción larga del README
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='MyShell-OpenVoice',
    version='0.1.0',
    description='Instant voice cloning by MyShell.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[
        'text-to-speech',
        'tts',
        'voice-clone',
        'zero-shot-tts',
        'voice-synthesis',
        'audio-processing'
    ],
    url='https://github.com/myshell-ai/OpenVoice',
    project_urls={
        'Documentation': 'https://github.com/myshell-ai/OpenVoice/blob/main/docs/USAGE.md',
        'Changes': 'https://github.com/myshell-ai/OpenVoice/releases',
        'Code': 'https://github.com/myshell-ai/OpenVoice',
        'Issue tracker': 'https://github.com/myshell-ai/OpenVoice/issues',
    },
    author='MyShell',
    author_email='ethan@myshell.ai',
    license='MIT License',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        # Audio processing
        'librosa>=0.10.0',
        'soundfile>=0.12.0',
        'pydub>=0.25.1',
        'soxr>=0.3.0',
        
        # Speech recognition and processing
        'faster-whisper>=1.0.0',
        'whisper-timestamped>=1.15.0',
        'wavmark>=0.0.3',
        
        # Text processing
        'numpy>=1.21.0',
        'inflect>=7.0.0',
        'unidecode>=1.3.0',
        
        # Chinese text processing
        'pypinyin>=0.50.0',
        'cn2an>=0.5.22',
        'jieba>=0.42.1',
        
        # Web interface
        'gradio>=4.0.0,<6.0.0',
        
        # Language detection
        'langid>=1.1.6',
        
        # Audio/Video processing
        'av>=10.0.0',
        
        # Additional utilities
        'python-dotenv>=1.0.0',
        'resampy>=0.4.0',
        
        # Note: eng_to_ipa is optional with fallback in code
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'all': [
            'eng_to_ipa>=0.0.2',  # Optional, has fallback in code
            'openai>=1.0.0',      # Optional for Whisper API
            'ctranslate2>=4.0.0', # Optional for faster-whisper acceleration
            'onnxruntime>=1.14.0', # Optional for faster-whisper
        ],
        'gpu': [
            'torch>=2.0.0',
            'torchaudio>=2.0.0',
        ],
        'cpu': [
            'torch>=2.0.0+cpu',
            'torchaudio>=2.0.0+cpu',
        ]
    },
    entry_points={
        'console_scripts': [
            'openvoice=openvoice.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'openvoice': [
            'checkpoints/**/*',
            'resources/**/*',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    zip_safe=False
)
