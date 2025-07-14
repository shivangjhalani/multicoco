#!/bin/bash
# Fix CUDA OOM issues in MultiCoCo training
# This script applies memory optimization strategies to prevent OOM errors

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}  MultiCoCo Memory Optimization  ${NC}"
echo -e "${BLUE}==================================${NC}"

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}=== $1 ===${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
print_section "Checking Dependencies"

MISSING_DEPS=0
for cmd in python3 nvidia-smi; do
    if ! command_exists $cmd; then
        echo -e "${RED}❌ $cmd is not installed${NC}"
        MISSING_DEPS=1
    else
        echo -e "${GREEN}✓ $cmd is installed${NC}"
    fi
done

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${RED}Some dependencies are missing. Please install them.${NC}"
    exit 1
fi

# Check GPU status
print_section "GPU Status Before Optimization"
if command_exists nvidia-smi; then
    nvidia-smi
else
    echo -e "${YELLOW}nvidia-smi not found. Skipping GPU check.${NC}"
fi

# Apply memory optimization environment variables
print_section "Setting Memory Optimization Variables"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo -e "${GREEN}✓ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True${NC}"

# Create a modified version of the config file with memory optimizations
apply_config_optimizations() {
    local config_file=$1
    local output_file="${config_file%.yaml}_optimized.yaml"
    
    echo -e "${BLUE}Creating optimized config: $output_file${NC}"
    
    # Read batch size and check if it needs adjustment
    local batch_size=$(grep -E "^batch_size:" "$config_file" | awk '{print $2}')
    
    if [ -z "$batch_size" ]; then
        batch_size=16  # Default if not specified
    fi
    
    # For multimodal models, we recommend batch size of 4-8
    local recommended_batch_size=8
    local grad_accum=2
    
    if [ "$batch_size" -gt "$recommended_batch_size" ]; then
        grad_accum=$(($batch_size / $recommended_batch_size))
        if [ $grad_accum -lt 1 ]; then
            grad_accum=1
        fi
        batch_size=$recommended_batch_size
    fi
    
    # Create optimized config
    cp "$config_file" "$output_file"
    
    # Update batch size
    if grep -q "^batch_size:" "$output_file"; then
        sed -i "s/^batch_size:.*$/batch_size: $batch_size/" "$output_file"
    else
        echo "batch_size: $batch_size" >> "$output_file"
    fi
    
    # Update gradient accumulation
    if grep -q "^gradient_accumulation_steps:" "$output_file"; then
        sed -i "s/^gradient_accumulation_steps:.*$/gradient_accumulation_steps: $grad_accum/" "$output_file"
    else
        echo "gradient_accumulation_steps: $grad_accum" >> "$output_file"
    fi
    
    # Enable gradient checkpointing
    if grep -q "^gradient_checkpointing:" "$output_file"; then
        sed -i "s/^gradient_checkpointing:.*$/gradient_checkpointing: true/" "$output_file"
    else
        echo "gradient_checkpointing: true" >> "$output_file"
    fi
    
    # Set fp16 or bf16 training
    if grep -q "^bf16:" "$output_file"; then
        sed -i "s/^bf16:.*$/bf16: true/" "$output_file"
    else
        echo "bf16: true" >> "$output_file"
    fi
    
    echo -e "${GREEN}✓ Created optimized config file: $output_file${NC}"
    echo -e "${GREEN}✓ Set batch_size: $batch_size${NC}"
    echo -e "${GREEN}✓ Set gradient_accumulation_steps: $grad_accum${NC}"
    echo -e "${GREEN}✓ Enabled gradient_checkpointing${NC}"
    echo -e "${GREEN}✓ Enabled bf16 precision${NC}"
    
    echo -e "${BLUE}Optimized config ready to use${NC}"
    
    # Return the name of the optimized config
    echo "$output_file"
}

# Run diagnostic
run_diagnostic() {
    print_section "Running Diagnostics"
    
    # Check if Python script exists
    if [ ! -f "debug_memory_issue.py" ]; then
        echo -e "${RED}❌ debug_memory_issue.py not found${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Running memory diagnostics...${NC}"
    python3 debug_memory_issue.py --check-env-only
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Diagnostic failed${NC}"
        return 1
    else
        echo -e "${GREEN}✓ Diagnostic completed${NC}"
    fi
}

# Main execution logic
print_section "Memory Optimization Steps"

echo -e "1. Applied memory optimization environment variables"
echo -e "2. Running diagnostics to check environment"
run_diagnostic

# Check for config file
CONFIG_FILE=""
if [ $# -gt 0 ] && [ -f "$1" ]; then
    CONFIG_FILE="$1"
    echo -e "3. Optimizing config file: $CONFIG_FILE"
    OPTIMIZED_CONFIG=$(apply_config_optimizations "$CONFIG_FILE")
    
    # Suggest command to run
    echo -e "\n${GREEN}================== RECOMMENDED COMMAND ==================${NC}"
    echo -e "Run your training with the optimized config using:"
    echo -e "${BLUE}PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nnodes 1 --nproc_per_node 1 run.py $OPTIMIZED_CONFIG${NC}"
    echo -e "${GREEN}=======================================================${NC}"
else
    echo -e "${YELLOW}No config file provided. Skipping config optimization.${NC}"
    echo -e "\n${GREEN}================== RECOMMENDED COMMAND ==================${NC}"
    echo -e "Run your training with memory optimizations using:"
    echo -e "${BLUE}PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nnodes 1 --nproc_per_node 1 run.py YOUR_CONFIG_FILE.yaml${NC}"
    echo -e "${GREEN}=======================================================${NC}"
fi

print_section "Additional Recommendations"
echo -e "1. ${YELLOW}Clear GPU cache before running:${NC} Execute 'nvidia-smi -r' as root"
echo -e "2. ${YELLOW}Close other GPU applications${NC} before training"
echo -e "3. ${YELLOW}Monitor memory usage${NC} with 'nvidia-smi -l 1' in another terminal"
echo -e "4. ${YELLOW}Consider system reboot${NC} if memory fragmentation persists"

echo -e "\n${BLUE}Memory optimization completed. Good luck with your training!${NC}"
