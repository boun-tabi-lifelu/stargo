#!/bin/bash
set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATA_DIR="datasets"
DATASET="both"
DEVICE="cuda"
SKIP_DOWNLOAD=false
SKIP_GO_EMBS=false
SKIP_RESIDUE_EMBS=false
SKIP_ZERO_SHOT=false

# Function to print colored messages
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-go-embs)
            SKIP_GO_EMBS=true
            shift
            ;;
        --skip-residue-embs)
            SKIP_RESIDUE_EMBS=true
            shift
            ;;
        --skip-zero-shot)
            SKIP_ZERO_SHOT=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-dir DIR          Directory for datasets (default: datasets)"
            echo "  --dataset TYPE          Dataset to prepare: pfresgo, deepgozero, or both (default: both)"
            echo "  --device DEVICE         Compute device: cuda, cpu, or mps (default: cuda)"
            echo "  --skip-download         Skip dataset download step"
            echo "  --skip-go-embs          Skip GO embedding generation"
            echo "  --skip-residue-embs     Skip residue embedding generation"
            echo "  --skip-zero-shot        Skip zero-shot data preparation"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             STARGO Data Preparation Pipeline                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Dataset: $DATASET"
echo "  Device: $DEVICE"
echo ""


# Step 1: Download datasets
if [ "$SKIP_DOWNLOAD" = false ]; then
    print_step "Step 1: Downloading datasets"
    (set -x; python bin/download_data.py --data-dir "$DATA_DIR" --dataset "$DATASET")
    print_success "Dataset download complete"
else
    print_warning "Skipping dataset download"
fi

# Step 2: Generate GO embeddings
if [ "$SKIP_GO_EMBS" = false ]; then
    print_step "Step 2: Generating GO embeddings"

    if [ "$DATASET" = "pfresgo" ] || [ "$DATASET" = "both" ]; then
        # Check if embeddings already exist
        if [ -f "embeddings/go-basic-2020-06-01.stargo.npy" ]; then
            print_warning "PFresGO GO embeddings already exist, skipping generation"
        else
            print_step "  Generating PFresGO GO embeddings (2020-06-01)..."
            (set -x; python bin/generate_go_embs.py --go-date 2020-06-01 --edition basic --device "$DEVICE")
        fi

        # Copy embeddings to dataset directory if they exist
        if [ -f "embeddings/go-basic-2020-06-01.stargo.npy" ]; then
            print_step "  Copying PFresGO embeddings..."
            (set -x; cp embeddings/go-basic-2020-06-01.stargo.npy "$DATA_DIR/pfresgo/ontology.embeddings.npy")
            print_success "PFresGO stargo embeddings copied"
        else
            print_warning "PFresGO stargo embeddings not found, skipping copy"
        fi

        if [ -f "embeddings/go-basic-2020-06-01.sbert.npy" ]; then
            (set -x; cp embeddings/go-basic-2020-06-01.sbert.npy "$DATA_DIR/pfresgo/ontology.sbert-embeddings.npy")
            print_success "PFresGO sbert embeddings copied"
        else
            print_warning "PFresGO sbert embeddings not found, skipping copy"
        fi
    fi

    if [ "$DATASET" = "deepgozero" ] || [ "$DATASET" = "both" ]; then
        # Check if embeddings already exist
        if [ -f "embeddings/go-basic-2021-11-16.stargo.npy" ]; then
            print_warning "DeepGOZero GO embeddings already exist, skipping generation"
        else
            print_step "  Generating DeepGOZero GO embeddings (2021-11-16)..."
            (set -x; python bin/generate_go_embs.py --go-date 2021-11-16 --edition basic --device "$DEVICE")
        fi

        # Copy embeddings to dataset directory if they exist
        if [ -f "embeddings/go-basic-2021-11-16.stargo.npy" ]; then
            print_step "  Copying DeepGOZero embeddings..."
            (set -x; cp embeddings/go-basic-2021-11-16.stargo.npy "$DATA_DIR/deepgozero/ontology.embeddings.npy")
            print_success "DeepGOZero embeddings copied"
        else
            print_warning "DeepGOZero embeddings not found, skipping copy"
        fi
    fi
else
    print_warning "Skipping GO embedding generation"
fi

# Step 3: Generate residue embeddings
if [ "$SKIP_RESIDUE_EMBS" = false ]; then
    print_step "Step 3: Generating residue embeddings (ProtT5)"

    if [ "$DATASET" = "pfresgo" ] || [ "$DATASET" = "both" ]; then
        print_step "  Generating PFresGO residue embeddings... (this may take a long time)"
        (set -x; python bin/generate_residue_embs.py \
            --data-dir "$DATA_DIR/pfresgo" \
            --dataset-type pfresgo \
            --ontology all \
            --output-file "$DATA_DIR/pfresgo/per_residue_embeddings.h5")
        print_success "PFresGO residue embeddings generated"
    fi

    if [ "$DATASET" = "deepgozero" ] || [ "$DATASET" = "both" ]; then
        print_step "  Generating DeepGOZero residue embeddings... (this may take a long time)"
        (set -x; python bin/generate_residue_embs.py \
            --data-dir "$DATA_DIR/deepgozero" \
            --dataset-type pugo \
            --ontology all \
            --output-file "$DATA_DIR/deepgozero/per_residue_embeddings.h5")
        print_success "DeepGOZero residue embeddings generated"
    fi
else
    print_warning "Skipping residue embedding generation"
fi

# Step 4: Prepare zero-shot data (DeepGOZero only)
if [ "$SKIP_ZERO_SHOT" = false ]; then
    if [ "$DATASET" = "deepgozero" ] || [ "$DATASET" = "both" ]; then
        print_step "Step 4: Preparing zero-shot evaluation data (DeepGOZero)"
        (set -x; python bin/prepare_zero_shot_data.py --subontology all)
        print_success "Zero-shot data preparation complete"
    else
        print_warning "Skipping zero-shot data preparation (DeepGOZero only)"
    fi
else
    print_warning "Skipping zero-shot data preparation"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  Data Preparation Complete!                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
print_success "All datasets are ready for training!"
echo ""
echo "Next steps:"
echo "  - Train models using: python bin/train.py -c configs/CONFIG.toml"
echo "  - See README.md for training examples"

