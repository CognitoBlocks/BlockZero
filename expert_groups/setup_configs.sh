#!/bin/bash
# Quick setup script for expert group configs
# Usage: ./setup_configs.sh

set -e

cd "$(dirname "$0")"

echo "Setting up expert group configurations..."

# Check if template exists
if [ ! -f "config.template.yaml" ]; then
    echo "Error: config.template.yaml not found!"
    exit 1
fi

# Function to create config from template
create_config() {
    local group=$1
    local id=$2
    local dataset_name=$3
    local dataset_class=$4
    local data_dir=${5:-null}

    echo "Creating config for $group..."

    # Copy template
    cp config.template.yaml "$group/config.yaml"

    # Update values using sed (use | as delimiter to handle / in values)
    sed -i.bak "s|dataset_name: \"merged_math\"|dataset_name: \"$dataset_name\"|" "$group/config.yaml"
    sed -i.bak "s|dataset_class: \"expert_groups.exp_math.dataset:MergedMathDataset\"|dataset_class: $dataset_class|" "$group/config.yaml"
    sed -i.bak "s|expert_group_id: 0|expert_group_id: $id|" "$group/config.yaml"
    sed -i.bak "s|expert_group_name: \"exp_math\"|expert_group_name: \"$group\"|" "$group/config.yaml"

    if [ "$data_dir" != "null" ]; then
        sed -i.bak "s|data_dir: null|data_dir: \"$data_dir\"|" "$group/config.yaml"
    fi

    # Remove backup files
    rm "$group/config.yaml.bak"

    echo "✓ Created $group/config.yaml"
}

# Create configs for each expert group (note: colon separates module from class)
create_config "exp_math" 0 "merged_math" "\"expert_groups.exp_math.dataset:MergedMathDataset\""
create_config "exp_agentic" 1 "merged_agentic" "\"expert_groups.exp_agentic.dataset:MergedAgenticDataset\""
create_config "exp_planning" 2 "merged_planning" "\"expert_groups.exp_planning.dataset:MergedPlanningDataset\""
create_config "exp_dummy" 99 "allenai/c4" "null" "en"

echo ""
echo "✓ All expert group configs created successfully!"
echo ""
echo "Files created:"
ls -1 exp_*/config.yaml

echo ""
echo "Next steps:"
echo "1. Review and customize configs if needed"
echo "2. Generate miner config: python mycelia/shared/config.py --get_template miner --coldkey_name <name> --hotkey_name <name> --run_name <name>"
