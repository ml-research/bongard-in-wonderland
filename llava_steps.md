# Setup LLaVa Models

## Build container
```bash
cd docker
docker compose build
docker compose up -d
docker exec -it bongard bash
```

## Start server (LLaVA 1.6)
```bash
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-34b --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --chat-template chatml-llava --port 10000
```

## Start server (LLaVA 1.5)
```bash
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-13b --tokenizer-path llava-hf/llava-1.5-13b-hf --port 10000
```

Open second container shell and execute scripts there.