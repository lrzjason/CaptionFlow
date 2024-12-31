### CaptionFlow: High-Fidelity Detailed Caption Generation

## Introduction

CaptionFlow is a tool designed to generate high-fidelity detailed captions for images using the Mistral API. Inspired by the paper "Visual Fact Checker: Enabling High-Fidelity Detailed Caption Generation," this project provides a user-friendly GUI to process batches of images and save captions. We acknowledge the work of SmilingWolf's wd14 tagger, which served as inspiration for this project.

## Installation

### Prerequisites

- Python 3.x
- Dependencies listed in `requirements.txt`

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/lrzjason/CaptionFlow.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Setting up api.json

Create an `api.json` file in the root directory with the following structure:

```json
{
  "MISTRAL_API_KEY": "your_api_key_here"
}
```

### Understanding config.json

`config.json` stores input configurations for future use. Default values are provided if `config.json` is missing.

## Usage

### Launching the GUI

Run the following command to launch the Gradio app:

```bash
python ui_gradio.py
```

Access the GUI through the provided local link.

### Using the Input Fields

- **Directory Path**: Input the path to the directory containing images.
- **Prefix**: Add a prefix to the generated captions.
- **API Key (Optional)**: Enter your Mistral API key if not set in `api.json`.
- **Include Additional CHI Types**: Select CHI types to include in caption generation.
- **Drop Rate for CHI**: Adjust the drop rate for CHI types to influence caption generation.

### Viewing Output

Captions are saved in the image directory with the specified prefix and are displayed in the GUI.

## Advanced Options

- **Customizing CHI Types**: Exclude certain CHI types by selecting them. Default all CHI included.
- **Adjusting Drop Rates**: Fine-tune the drop rate for CHI types to influence caption generation.

## Contributing

Contributions are welcome. Please submit issues or pull requests following standard GitHub practices.

## License

This project is licensed under the Apache-2.0 License.

## Contact

For inquiries, please contact the maintainer at [lrzjason@gmail.com].

## Acknowledgments
Special thanks to deepseek v3 help to construct the ui and the readme.

Thank you to SmilingWolf for the inspiration from wd14 tagger.

Special thanks to the authors of "Visual Fact Checker: Enabling High-Fidelity Detailed Caption Generation" for their insightful work.