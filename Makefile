ATTACK_DIR := dataset/processed_files/attack_data/
BENIGN_DIR := dataset/processed_files/benign_data/
URL := http://205.174.165.80/IOTDataset/Datasense/Dataset/

all: build 

$(ATTACK_DIR):
	wget -r -np -nH --cut-dirs=3 -R index.html $(URL)$(ATTACK_DIR)
	rm index.html*

$(BENIGN_DIR):
	wget -r -np -nH --cut-dirs=3 -R index.html $(URL)$(BENIGN_DIR)
	rm index.html*

build: $(ATTACK_DIR) $(BENIGN_DIR)
	uv run python tools/unpack_dataset.py ./dataset/processed_files/attack_data/ ./data/attack_data/
	uv run python tools/unpack_dataset.py ./dataset/processed_files/benign_data/ ./data/benign_data/

.PHONY: download
download: $(ATTACK_DIR) $(BENIGN_DIR)
