


<h3><a href="">Context Cascade Compression: Exploring the Upper Limits of Text Compression</a></h3>

<a href="https://huggingface.co/liufanfanlff/C3-Context-Cascade-Compression"><img src="https://img.shields.io/badge/Huggingface-yellow"></a>
<a href="https://arxiv.org/abs/2511.15244"><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 


[Fanfan Liu](https://scholar.google.com/citations?user=LPaXZEUAAAAJ&hl=en), [Haibo Qiu](https://scholar.google.com/citations?user=O5gH5vkAAAAJ&hl=en)


<p align="center">
<img src="assets/8a5ce4ed-f1d5-4a3c-8718-3604cf3c3866.png" style="width: 1000px" align=center>
</p> 


## Release

- [2025/11/20]🔥🔥🔥 We open-source the codes, weights. The paper can be found in this [repo](https://github.com/liufanfanlff/C3-Context-Cascade-Compression/blob/main/C3.pdf).  
- [2025/11/20]🔥🔥🔥 We release the C3 model! 


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)


## Contents
- [Install](#install)
- [GOT Weights](#got-weights)
- [Benchmarks](#benchmarks)
- [Demo](#demo)




## Install

1. Clone this repository and navigate to the GOT folder
```bash
git clone https://github.com/liufanfanlff/C3-Context-Cascade-Compression.git
```
2. Install Package
```Shell
conda create -n got python=3.10 -y
conda activate got
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.49.0 transformers-stream-generator==0.0.5

```

## GOT Weights
- [Huggingface](https://huggingface.co/liufanfanlff/C3-Context-Cascade-Compression)


## Benchmarks
- [Fox](https://github.com/ucaslcl/Fox)

## Demo
Transformers:
```Shell
from transformers import AutoModel, AutoTokenizer

model_name = '../liufanfanlff/C3-Context-Cascade-Compression'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name , trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()
prompt = 'Repeat the text: '
context = "帝高阳之苗裔兮，朕皇考曰伯庸。摄提贞于孟陬兮，"
#context = "lfflfflfflfflfflfflfflfflff"
outputs = model.chat(tokenizer, context, prompt)
print ("Repeat the text: ",outputs)
```
or you can:
```Shell
python3 /C3-master/C3-hf/run_c3.py
 ```

viz
<p align="center">
<img src="assets/et.png" style="width: 1000px" align=center>
</p>


## Contact

Don't hesitate to contact me by email, liufanfan19@mails.ucas.ac.cn, if you have any questions.

## Acknowledgement
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR): the idea originated from reconsideration of this work.
- [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0): the code was adapted from GOT-OCR2.0.
- [Qwen](https://github.com/QwenLM/Qwen): the LLM base model of C3.



## Citation
```bibtex
@article{liu2025context,
  title={Context Cascade Compression: Exploring the Upper Limits of Text Compression},
  author={Liu, Fanfan and Qiu, Haibo},
  journal={arXiv preprint arXiv:2511.15244},
  year={2025}
}



