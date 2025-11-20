<h3><a href="">Context Cascade Compression: Exploring the Upper Limits of Text Compression</a></h3>

<a href="https://huggingface.co/liufanfanlff/C3-Context-Cascade-Compression"><img src="https://img.shields.io/badge/Huggingface-yellow"></a>
<!-- <a href="https://modelscope.cn/models/stepfun-ai/GOT-OCR2_0"><img src="https://img.shields.io/badge/Modelscope-red"></a> -->
<a href="https://arxiv.org/abs/2511.15244"><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 


[Fanfan Liu](https://scholar.google.com/citations?user=LPaXZEUAAAAJ&hl=en), [Haibi Qiu](https://scholar.google.com/citations?user=O5gH5vkAAAAJ&hl=en)
<!-- <p align="center">
<img src="assets/got_logo.png" style="width: 200px" align=center>
</p> -->


## Release
<!-- - [2025/2/1] 🚀🚀🚀 GOT-OCR2.0 is merged to [Huggingface-transformers](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)/[space](https://huggingface.co/spaces/yonigozlan/GOT-OCR-Transformers). It supports inference batched. Thanks to the MLE of Huggingface [Yoni](https://github.com/yonigozlan).
- [2024/12/24] 🔥🔥🔥 My new work on system-2 perception is released [slow-perception](https://github.com/Ucas-HaoranWei/Slow-Perception).
- [2024/12/18] 🚀🚀🚀 GOT-OCR2.0 is supported in [PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/GOT_OCR_2_0) by Paddle Team. Thanks for the Paddle team!
- [2024/12/8] 🔥🔥🔥 The model download has exceeded 1M on [Huggingface](https://huggingface.co/stepfun-ai/GOT-OCR2_0).
- [2024/12/5] The seven wechat [group](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/blob/main/assets/Wechat7.jpg).
- [2024/11/4] The six wechat [group](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/blob/main/assets/wechat6-2.jpg).
- [2024/10/24] The previous four wechat groups are full, so we created a fifth [group](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/blob/main/assets/wechat5.png).
- [2024/10/11] Too many friends want to join the wechat group, so we created a fourth [group](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/blob/main/assets/wechat4.jpg).
- [2024/10/2] [onnx](https://github.com/BaofengZan/GOT-OCRv2-onnx) and [mnn](https://github.com/BaofengZan/mnn-llm-GOT-OCR2.0) versions of GOT-OCR2.0.
- [2024/9/29]🔥🔥🔥 The community has implemented the first version of [llama_cpp_inference](https://github.com/1694439208/GOT-OCR-Inference).
- [2024/9/24]🔥🔥🔥 Support [ms-swift](https://github.com/modelscope/ms-swift/issues/2122) quick [Fine-tune](#fine-tune) for your own data. 
- [2024/9/23]🔥🔥🔥 We release the official [Modelscope demo](https://modelscope.cn/studios/stepfun-ai/GOT_official_online_demo). Thanks very much for Modelscope providing the GPU resource.
- [2024/9/19]🔥🔥🔥 GOT-OCR2.0 achieves Huggingface trending #1.
- [2024/9/14]🔥🔥🔥 We release the official [demo](https://huggingface.co/spaces/ucaslcl/GOT_online). Thanks very much for Huggingface providing the GPU resource. 
- [2024/9/13]🔥🔥🔥 We release the [Huggingface](https://huggingface.co/ucaslcl/GOT-OCR2_0) deployment. 
-->
- [2024/9/03]🔥🔥🔥 We open-source the codes, weights. The paper can be found in this [repo](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/blob/main/GOT-OCR-2.0-paper.pdf).  
- [2025/9/03]🔥🔥🔥 We release the C3 model! 


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)




<!-- ## Community contributions
We encourage everyone to develop GOT applications based on this repo. Thanks for the following contributions :

[OpenVINO](https://github.com/can-gaa-hou/GOT-OCR2.0-OpenVINO)~ contributor: [@can-gaa-hou](https://github.com/can-gaa-hou)

[GGUF and Llama.cpp inference](https://github.com/MosRat/got.cpp)~ contributor: [@MosRat](https://github.com/MosRat)

[vllm reference](https://github.com/liunian-Jay/MU-GOT/blob/master/PDF_parsing/GOT/GOT/model/modeling_GOT_vllm.py) ~ contributor: [@Jay](https://github.com/liunian-Jay)

[onnx and mnn supports](https://github.com/BaofengZan/GOT-OCRv2-onnx) ~ contributor: [@BaofengZan](https://github.com/BaofengZan)

[llama_cpp inference](https://github.com/1694439208/GOT-OCR-Inference) ~ contributor: [@1694439208](https://github.com/1694439208)

[Colab of GOT](https://colab.research.google.com/drive/1nmiNciZ5ugQVp4rFbL9ZWpEPd92Y9o7p?usp=sharing)   ~      contributor: [@Zizhe Wang](https://github.com/PaperPlaneDeemo)

[CPU version of GOT](https://github.com/ElvisClaros/GOT-OCR2.0) ~ contributor: [@ElvisClaros](https://github.com/ElvisClaros)

[Online demo](https://huggingface.co/spaces/Tonic/GOT-OCR) ~ contributor: [@Joseph Pollack](https://huggingface.co/Tonic)

[Dokcer & client demo](https://github.com/QIN2DIM/GOT-OCR2.0) ~ contributor: [@QIN2DIM](https://github.com/QIN2DIM) 

[GUI of GOT](https://github.com/XJF2332/GOT-OCR-2-GUI) ~ contributor: [@XJF2332](https://github.com/XJF2332)  -->

## Contents
- [Install](#install)
- [GOT Weights](#got-weights)
- [Benchmarks](#benchmarks)
- [Demo](#demo)
- [Train](#train)
- [Fine-tune](#fine-tune)
- [Eval](#eval)




## Install
0. Our environment is cuda11.8+torch2.0.1
1. Clone this repository and navigate to the GOT folder
```bash
git clone https://github.com/Ucas-HaoranWei/GOT-OCR2.0.git

```
2. Install Package
```Shell
conda create -n got python=3.10 -y
conda activate got
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.7.3 --no-build-isolation

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


## Contact

Don't hesitate to contact me by email, liufanfan19@mails.ucas.ac.cn, if you have any questions.

## Acknowledgement
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR): The idea originated from reconsideration of this work.
- [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0): The code was adapted from GOT-OCR2.0.
- [Qwen](https://github.com/QwenLM/Qwen): the LLM base model of C3.



## Citation
```bibtex
@article{liu2025context,
  title={Context Cascade Compression: Exploring the Upper Limits of Text Compression},
  author={Liu, Fanfan and Qiu, Haibo},
  journal={arXiv preprint arXiv:2511.15244},
  year={2025}
}



