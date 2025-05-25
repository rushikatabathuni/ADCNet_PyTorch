import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataset import Inference_Dataset
import math
import sys
import os
from ESM2 import get_sequence_embedding

def normalize_dar_value(dar_value):
    """
    Normalize DAR value using the same statistics as training
    """
    mean_value = 3.86845977
    variance_value = 1.569108443
    std_deviation = variance_value**0.5
    dar_standardized = (dar_value - mean_value) / std_deviation
    dar_normalized = (dar_standardized - 0.8) / (12 - 0.8)
    return dar_normalized

def validate_inputs(heavy_seq, light_seq, antigen_seq, payload_smiles, linker_smiles, dar_value):
    """
    Validate user inputs
    """
    errors = []
    
    if len(heavy_seq) < 10:
        errors.append("Heavy chain sequence seems too short (minimum 10 amino acids)")
    if len(light_seq) < 10:
        errors.append("Light chain sequence seems too short (minimum 10 amino acids)")
    if len(antigen_seq) < 10:
        errors.append("Antigen sequence seems too short (minimum 10 amino acids)")
    
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    for seq_name, seq in [('Heavy', heavy_seq), ('Light', light_seq), ('Antigen', antigen_seq)]:
        invalid_chars = set(seq.upper()) - valid_aa
        if invalid_chars:
            errors.append(f"{seq_name} sequence contains invalid characters: {invalid_chars}")
    
    if not payload_smiles.strip():
        errors.append("Payload SMILES cannot be empty")
    if not linker_smiles.strip():
        errors.append("Linker SMILES cannot be empty")
    
    if dar_value <= 0 or dar_value > 10:
        errors.append("DAR value should be between 0 and 10")
    
    return errors

def single_sample_inference(heavy_sequence, light_sequence, antigen_sequence, 
                          payload_isosmiles, linker_isosmiles, dar_value, 
                          model_path='classification_weights/ADC_9.pth'):
    """
    Perform inference on a single ADC sample
    """
    
    num_layers = 6
    num_heads = 8
    d_model = 256
    dff = d_model * 2
    vocab_size = 18
    dense_dropout = 0.1
    addH = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        from model import PredictModel
        model = PredictModel(num_layers=num_layers,
                            d_model=d_model,
                            dff=dff, 
                            num_heads=num_heads, 
                            vocab_size=vocab_size,
                            dropout_rate=dense_dropout)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")
    
    print("\n🧬 Generating ESM embeddings...")
    heavy_embedding = get_sequence_embedding(heavy_sequence)
    light_embedding = get_sequence_embedding(light_sequence)
    antigen_embedding = get_sequence_embedding(antigen_sequence)
    
    dar_normalized = normalize_dar_value(dar_value)
    
    t1 = torch.tensor(heavy_embedding, dtype=torch.float32).unsqueeze(0)
    t2 = torch.tensor(light_embedding, dtype=torch.float32).unsqueeze(0)
    t3 = torch.tensor(antigen_embedding, dtype=torch.float32).unsqueeze(0)
    t4 = torch.tensor([[dar_normalized]], dtype=torch.float32)
    
    print("🧪 Processing molecular structures...")
    try:
        x1 = [payload_isosmiles]
        x2 = [linker_isosmiles]
        
        inference_dataset1 = Inference_Dataset(x1, addH=addH).get_data()
        inference_dataset2 = Inference_Dataset(x2, addH=addH).get_data()
        
        x1_batch, adjoin_matrix1_batch, smiles1, atom_list1 = next(iter(inference_dataset1))
        x2_batch, adjoin_matrix2_batch, smiles2, atom_list2 = next(iter(inference_dataset2))
        
        x1_tensor = x1_batch[0:1]
        adjoin_matrix1 = adjoin_matrix1_batch[0:1]
        x2_tensor = x2_batch[0:1]
        adjoin_matrix2 = adjoin_matrix2_batch[0:1]
        
        seq1 = (x1_tensor == 0).float()
        seq2 = (x2_tensor == 0).float()
        mask1 = seq1.unsqueeze(1).unsqueeze(1)
        mask2 = seq2.unsqueeze(1).unsqueeze(1)
        
        # Move tensors to device
        x1_tensor = x1_tensor.to(device)
        x2_tensor = x2_tensor.to(device)
        adjoin_matrix1 = adjoin_matrix1.to(device)
        adjoin_matrix2 = adjoin_matrix2.to(device)
        mask1 = mask1.to(device)
        mask2 = mask2.to(device)
        t1 = t1.to(device)
        t2 = t2.to(device)
        t3 = t3.to(device)
        t4 = t4.to(device)
    except Exception as e:
        raise RuntimeError(f"Error processing molecular structures: {str(e)}")
    
    print("🤖 Running model inference...")
    try:
        with torch.no_grad():
            logits = model(x1=x1_tensor, 
                          mask1=mask1, 
                          training=False,
                          adjoin_matrix1=adjoin_matrix1, 
                          x2=x2_tensor, 
                          mask2=mask2, 
                          adjoin_matrix2=adjoin_matrix2,
                          t1=t1,
                          t2=t2,
                          t3=t3,
                          t4=t4)
            
            probability = torch.sigmoid(logits).item()
            binary_prediction = 1 if probability >= 0.5 else 0
            confidence = abs(probability - 0.5) * 2
            
    except Exception as e:
        raise RuntimeError(f"Error during model inference: {str(e)}")
    
    return {
        'probability': probability,
        'binary_prediction': binary_prediction,
        'confidence': confidence
    }

def get_user_input():
    """
    Interactive command line input collection
    """
    print("=" * 80)
    print("🧬 ADCNet Single Sample Inference Tool")
    print("=" * 80)
    print("Please provide the following information for ADC prediction:")
    print()
    
    print("📝 Enter protein sequences (amino acid sequences):")
    heavy_seq = input("Heavy Chain Sequence: ").strip().upper()
    light_seq = input("Light Chain Sequence: ").strip().upper()
    antigen_seq = input("Antigen Sequence: ").strip().upper()
    
    print("\n🧪 Enter molecular information:")
    payload_smiles = input("Payload SMILES: ").strip()
    linker_smiles = input("Linker SMILES: ").strip()
    
    # Get DAR value with validation
    while True:
        try:
            dar_input = input("DAR (Drug-to-Antibody Ratio) value: ").strip()
            dar_value = float(dar_input)
            break
        except ValueError:
            print("❌ Please enter a valid number for DAR value.")
    
    return heavy_seq, light_seq, antigen_seq, payload_smiles, linker_smiles, dar_value

def display_results(result):
    """
    Display prediction results in a formatted way
    """
    print("\n" + "=" * 80)
    print("🎯 PREDICTION RESULTS")
    print("=" * 80)
    
    probability = result['probability']
    binary_pred = result['binary_prediction']
    confidence = result['confidence']
    
    print(f"📊 Prediction Probability: {probability:.4f}")
    print(f"🎯 Binary Prediction: {'POSITIVE' if binary_pred == 1 else 'NEGATIVE'} ({binary_pred})")
    print(f"🔍 Confidence Score: {confidence:.4f} ({confidence*100:.1f}%)")
    
    print("\n📋 INTERPRETATION:")
    if binary_pred == 1:
        print("✅ The ADC is predicted to be ACTIVE/EFFECTIVE")
    else:
        print("❌ The ADC is predicted to be INACTIVE/INEFFECTIVE")
    
    if confidence > 0.8:
        print("🔥 High confidence prediction")
    elif confidence > 0.6:
        print("🟡 Moderate confidence prediction")
    else:
        print("⚠️  Low confidence prediction - consider additional validation")
    
    print("=" * 80)

def main():
    """
    Main function for command line interface
    """
    try:
        heavy_seq, light_seq, antigen_seq, payload_smiles, linker_smiles, dar_value = get_user_input()
        
        print("\n🔍 Validating inputs...")
        errors = validate_inputs(heavy_seq, light_seq, antigen_seq, payload_smiles, linker_smiles, dar_value)
        
        if errors:
            print("\n❌ Input validation errors:")
            for error in errors:
                print(f"  • {error}")
            print("\nPlease fix the errors and try again.")
            return
        
        print("✅ All inputs are valid!")
        
        print("\n🚀 Starting inference...")
        result = single_sample_inference(
            heavy_sequence=heavy_seq,
            light_sequence=light_seq,
            antigen_sequence=antigen_seq,
            payload_isosmiles=payload_smiles,
            linker_isosmiles=linker_smiles,
            dar_value=dar_value
        )
        
        display_results(result)
        
        print("\nWould you like to make another prediction? (y/n): ", end="")
        if input().lower().startswith('y'):
            print("\n")
        else:
            print("Thank you for using ADCNet! 👋")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your inputs and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
