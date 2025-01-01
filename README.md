### CaptionFlow: High-Fidelity Detailed Caption Generation

## Introduction

CaptionFlow is a tool designed to generate high-fidelity detailed captions for images using the Mistral API.

Inspired by the paper "Visual Fact Checker: Enabling High-Fidelity Detailed Caption Generation" 

This project provides a user-friendly GUI to process batches of images and save captions. 

The each image is processed by multiple Complex Human Instructions (CHI) for different aspect captions.

For example: 
- CHI_BLUR would focus on the blurriness of the image.
- CHI_COLOR would focus on the color of the image.
- etc

After all CHI processed, a summary would be made based on all previous result and the caption would be refined as precise and accurate.

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
![alt text](https://github.com/lrzjason/CaptionFlow/blob/main/image/screenshot.png)

Access the GUI through the provided local link.

### Using the Input Fields

- **Directory Path**: Input the path to the directory containing images.
- **Prefix**: Add a prefix to the generated captions.
- **API Key (Optional)**: Enter your Mistral API key if not set in `api.json`.
- **Exclude CHI Types**: Select CHI types to exclude in caption generation. Default all CHI included.
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
- **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)  
- **Email**: lrzjason@gmail.com  
- **QQ Group**: 866612947  
- **Civitai**: [xiaozhijason](https://civitai.com/user/xiaozhijason)


## Sponsors me for more open source projects:
<div align="center">
  <table>
    <tr>
      <td align="center">
        <p>Buy me a coffee:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/bmc_qr.png" alt="Buy Me a Coffee QR" width="200" />
      </td>
      <td align="center">
        <p>WeChat:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/wechat.jpg" alt="WeChat QR" width="200" />
      </td>
    </tr>
  </table>
</div>

## Acknowledgments
Special thanks to deepseek v3 help to construct the ui and the readme.

Thank you to SmilingWolf for the inspiration from wd14 tagger.

Special thanks to the authors of "Visual Fact Checker: Enabling High-Fidelity Detailed Caption Generation" for their insightful work.
