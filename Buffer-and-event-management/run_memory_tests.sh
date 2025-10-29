#!/bin/bash

# MPS Memory Leak Fixes - Automated Test Runner for All Models
# This script runs the full test suite across all model types

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
EPOCHS=10
BATCHES_PER_EPOCH=500
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# All model types to test
MODEL_TYPES=("simple" "conv" "transformer" "lstm" "rnn" "gru" "vae" "resnet" "unet")

# Environment paths (MODIFY THESE TO MATCH YOUR SETUP)
BASELINE_ENV="./envs/torch_baseline"
FIXED_ENV="./envs/torch_PR_2"

# Results directory
RESULTS_DIR="test_results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}MPS Memory Leak Fixes - Comprehensive Test Suite${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Epochs: ${EPOCHS}"
echo "  Batches per epoch: ${BATCHES_PER_EPOCH}"
echo "  Total batches per model: $((EPOCHS * BATCHES_PER_EPOCH))"
echo "  Number of models: ${#MODEL_TYPES[@]}"
echo "  Models: ${MODEL_TYPES[*]}"
echo "  Timestamp: ${TIMESTAMP}"
echo ""
echo "Environments:"
echo "  Baseline: ${BASELINE_ENV}"
echo "  Fixed: ${FIXED_ENV}"
echo ""
echo "Results directory: ${RESULTS_DIR}/"
echo ""

# Check if environments exist
if [ ! -d "${BASELINE_ENV}" ]; then
    echo -e "${RED}ERROR: Baseline environment not found at ${BASELINE_ENV}${NC}"
    echo "Create it with: python3 -m venv ${BASELINE_ENV}"
    exit 1
fi

if [ ! -d "${FIXED_ENV}" ]; then
    echo -e "${RED}ERROR: Fixed environment not found at ${FIXED_ENV}${NC}"
    echo "Create it with: python3 -m venv ${FIXED_ENV}"
    exit 1
fi

# Function to check if MPS is available
check_mps() {
    python3 -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available'" 2>&1
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: MPS not available in current environment${NC}"
        return 1
    fi
    return 0
}

# Function to run test for a specific model
run_model_test() {
    local mode=$1
    local env_path=$2
    local model_type=$3
    local output_file=$4

    echo -e "${CYAN}────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}Testing ${model_type} model in ${mode} mode${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────${NC}"
    echo ""

    # Activate environment
    source "${env_path}/bin/activate"

    # Check MPS availability
    if ! check_mps; then
        deactivate
        return 1
    fi

    # Run test
    python3 test_memory_leak_fixes.py \
        --mode "${mode}" \
        --epochs ${EPOCHS} \
        --batches-per-epoch ${BATCHES_PER_EPOCH} \
        --model-type "${model_type}" \
        --output "${output_file}" 2>&1 | tee "${output_file%.json}.log"

    local exit_code=$?

    deactivate

    if [ $exit_code -ne 0 ]; then
        echo -e "${RED}ERROR: ${model_type} ${mode} test failed${NC}"
        return 1
    fi

    echo ""
    echo -e "${GREEN}✅ ${model_type} ${mode} test completed${NC}"
    echo ""

    return 0
}

# Parse command line arguments
SKIP_BASELINE=false
SKIP_FIXED=false
SKIP_COMPARE=false
SKIP_PLOT=false
SPECIFIC_MODELS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --skip-fixed)
            SKIP_FIXED=true
            shift
            ;;
        --skip-compare)
            SKIP_COMPARE=true
            shift
            ;;
        --skip-plot)
            SKIP_PLOT=true
            shift
            ;;
        --epochs)
            EPOCHS=$2
            shift 2
            ;;
        --batches-per-epoch)
            BATCHES_PER_EPOCH=$2
            shift 2
            ;;
        --models)
            # Read comma-separated model list
            IFS=',' read -ra SPECIFIC_MODELS <<< "$2"
            shift 2
            ;;
        --baseline-env)
            BASELINE_ENV=$2
            shift 2
            ;;
        --fixed-env)
            FIXED_ENV=$2
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-baseline         Skip baseline tests"
            echo "  --skip-fixed            Skip fixed tests"
            echo "  --skip-compare          Skip comparison"
            echo "  --skip-plot             Skip plot generation"
            echo "  --epochs N              Number of epochs (default: 5)"
            echo "  --batches-per-epoch N   Batches per epoch (default: 100)"
            echo "  --models MODEL1,MODEL2  Test specific models (comma-separated)"
            echo "  --baseline-env PATH     Path to baseline environment"
            echo "  --fixed-env PATH        Path to fixed environment"
            echo "  --help                  Show this help message"
            echo ""
            echo "Available models: ${MODEL_TYPES[*]}"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Test all models"
            echo "  $0 --models simple,conv,lstm          # Test specific models"
            echo "  $0 --skip-baseline                    # Only run fixed tests"
            echo "  $0 --epochs 3 --batches-per-epoch 50  # Quick test"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Use specific models if provided, otherwise use all
if [ ${#SPECIFIC_MODELS[@]} -gt 0 ]; then
    MODELS_TO_TEST=("${SPECIFIC_MODELS[@]}")
    echo "Testing specific models: ${MODELS_TO_TEST[*]}"
else
    MODELS_TO_TEST=("${MODEL_TYPES[@]}")
    echo "Testing all models: ${MODELS_TO_TEST[*]}"
fi
echo ""

# Start tests
START_TIME=$(date +%s)

# Arrays to track results
declare -a BASELINE_FILES
declare -a FIXED_FILES
declare -a MODELS_TESTED

# Test each model
for model in "${MODELS_TO_TEST[@]}"; do
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}MODEL: ${model}${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""

    BASELINE_FILE="${RESULTS_DIR}/baseline_${model}_${TIMESTAMP}.json"
    FIXED_FILE="${RESULTS_DIR}/fixed_${model}_${TIMESTAMP}.json"

    # Baseline test
    if [ "$SKIP_BASELINE" = false ]; then
        echo -e "${YELLOW}[1/2] Running BASELINE test for ${model}...${NC}"
        echo ""
        if run_model_test "baseline" "${BASELINE_ENV}" "${model}" "${BASELINE_FILE}"; then
            BASELINE_FILES+=("${BASELINE_FILE}")
            echo "Waiting 10 seconds for system to settle..."
            sleep 10
        else
            echo -e "${RED}Baseline test failed for ${model}, skipping to next model${NC}"
            continue
        fi
    else
        echo -e "${YELLOW}Skipping baseline test for ${model}${NC}"
    fi

    # Fixed test
    if [ "$SKIP_FIXED" = false ]; then
        echo -e "${YELLOW}[2/2] Running FIXED test for ${model}...${NC}"
        echo ""
        if run_model_test "fixed" "${FIXED_ENV}" "${model}" "${FIXED_FILE}"; then
            FIXED_FILES+=("${FIXED_FILE}")
            MODELS_TESTED+=("${model}")
            echo "Waiting 10 seconds for system to settle..."
            sleep 10
        else
            echo -e "${RED}Fixed test failed for ${model}${NC}"
        fi
    else
        echo -e "${YELLOW}Skipping fixed test for ${model}${NC}"
    fi

    echo ""
done

# Comparison and plotting
if [ ${#MODELS_TESTED[@]} -gt 0 ]; then
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}COMPARISONS AND PLOTS${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""

    # Use fixed environment for comparison/plotting
    source "${FIXED_ENV}/bin/activate"

    for i in "${!MODELS_TESTED[@]}"; do
        model="${MODELS_TESTED[$i]}"
        baseline_file="${BASELINE_FILES[$i]}"
        fixed_file="${FIXED_FILES[$i]}"

        echo -e "${CYAN}Processing ${model}...${NC}"

        # Compare
        if [ "$SKIP_COMPARE" = false ] && [ -f "${baseline_file}" ] && [ -f "${fixed_file}" ]; then
            echo "Comparing results..."
            python3 test_memory_leak_fixes.py \
                --mode compare \
                --baseline "${baseline_file}" \
                --fixed "${fixed_file}" \
                2>&1 | tee "${RESULTS_DIR}/comparison_${model}_${TIMESTAMP}.txt"
            echo ""
        fi

        # Plot
        if [ "$SKIP_PLOT" = false ] && [ -f "${baseline_file}" ] && [ -f "${fixed_file}" ]; then
            echo "Generating plot..."
            python3 test_memory_leak_fixes.py \
                --mode plot \
                --baseline "${baseline_file}" \
                --fixed "${fixed_file}"

            # Move plot to results directory
            PLOT_FILE=$(ls -t memory_leak_fixes_comparison_*.png 2>/dev/null | head -1)
            if [ -n "${PLOT_FILE}" ]; then
                mv "${PLOT_FILE}" "${RESULTS_DIR}/plot_${model}_${TIMESTAMP}.png"
                echo -e "${GREEN}Plot saved: ${RESULTS_DIR}/plot_${model}_${TIMESTAMP}.png${NC}"
            fi
            echo ""
        fi
    done

    deactivate
fi

# Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}TEST SUITE COMPLETE!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo -e "${GREEN}Total time: ${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo "Models tested: ${#MODELS_TESTED[@]}"
echo "  ${MODELS_TESTED[*]}"
echo ""
echo "Results directory: ${RESULTS_DIR}/"
echo ""
echo "Generated files:"
for model in "${MODELS_TESTED[@]}"; do
    echo -e "  ${model}:"
    [ -f "${RESULTS_DIR}/baseline_${model}_${TIMESTAMP}.json" ] && echo -e "    ${GREEN}✓${NC} baseline_${model}_${TIMESTAMP}.json"
    [ -f "${RESULTS_DIR}/fixed_${model}_${TIMESTAMP}.json" ] && echo -e "    ${GREEN}✓${NC} fixed_${model}_${TIMESTAMP}.json"
    [ -f "${RESULTS_DIR}/comparison_${model}_${TIMESTAMP}.txt" ] && echo -e "    ${GREEN}✓${NC} comparison_${model}_${TIMESTAMP}.txt"
    [ -f "${RESULTS_DIR}/plot_${model}_${TIMESTAMP}.png" ] && echo -e "    ${GREEN}✓${NC} plot_${model}_${TIMESTAMP}.png"
done

echo ""
echo "Next steps:"
echo "  1. Review comparison outputs in ${RESULTS_DIR}/"
echo "  2. Check plots for visual confirmation"
echo "  3. Look for >60% memory reduction across models"
echo ""
echo -e "${GREEN}Done!${NC}"
