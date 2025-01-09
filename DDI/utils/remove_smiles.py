import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem, RDLogger
import sys

RDLogger.DisableLog('rdApp.*')
tqdm.pandas()

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 'X'
        else:
            return mol
    except Exception as e:
        print(f"Error in SMILES conversion: {e}")
        return 'X'

def process_smiles_data(folder_path, utils_path):
    print(f"Processing folder: {folder_path}")  # 중간 상태 확인
    folder = Path(folder_path)
    utils_dir = Path(utils_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)
    if not utils_dir.exists():
        print(f"Error: Folder '{utils_dir}' does not exist.")
        sys.exit(1)
    
    # 파일 경로 지정
    trn_path, val_path, tst_path = folder / 'train.csv', folder / 'valid.csv', folder / 'test.csv'
    
    # 필요한 파일 존재 여부 확인
    if not trn_path.exists() or not val_path.exists() or not tst_path.exists():
        print(f"Error: One or more required files ('train.csv', 'valid.csv', 'test.csv') are missing in '{folder_path}'.")
        sys.exit(1)

    try:
        # 파일 읽기
        trn = pd.read_csv(trn_path)
        val = pd.read_csv(val_path)
        tst = pd.read_csv(tst_path)

        print(f"Read train: {len(trn)}, valid: {len(val)}, test: {len(tst)}")

        # Split 정보 추가
        trn['split'] = 'train'
        val['split'] = 'valid'
        tst['split'] = 'test'

        # 데이터 합치기
        total = pd.concat([trn, val, tst]).reset_index(drop=True)
        total_len = len(total)
        print(f"Combined total records: {total_len}")

        # SMILES 처리
        total['mol1'] = total['Drug1'].progress_apply(smiles_to_mol)
        total['mol2'] = total['Drug2'].progress_apply(smiles_to_mol)
        total = total[(total['mol1'] != 'X') & (total['mol2'] != 'X')].drop(columns=['mol1', 'mol2'])
        print(f"Removed invalid SMILES: {total_len - len(total)}")

        # 고유 약물 데이터셋 생성
        all_drugs = pd.concat([
            total[['Drug1_ID', 'Drug1']].rename(columns={'Drug1_ID': 'drug_id', 'Drug1': 'smiles'}),
            total[['Drug2_ID', 'Drug2']].rename(columns={'Drug2_ID': 'drug_id', 'Drug2': 'smiles'})
        ])
        drugs = all_drugs.drop_duplicates(subset=['drug_id', 'smiles'], keep='first').reset_index(drop=True)
        print(f"Unique drugs: {len(drugs)}")

        # 저장 경로 설정
        new_fd = folder / 'clean_smi'
        new_fd.mkdir(parents=True, exist_ok=True)

        # utils 폴더에 저장
        utils_dir.mkdir(parents=True, exist_ok=True)

        # 결과 저장
        total.to_csv(utils_dir / 'all_pairs.csv', index=False, header=True)
        drugs.to_csv(utils_dir / 'all_smiles.csv', index=False, header=True)

        # train/valid/test 파일은 folder_path의 clean_smi 폴더에 저장
        total[total['split'] == 'train'].to_csv(new_fd / 'train.csv', index=False, header=True)
        total[total['split'] == 'valid'].to_csv(new_fd / 'valid.csv', index=False, header=True)
        total[total['split'] == 'test'].to_csv(new_fd / 'test.csv', index=False, header=True)
        print(f"Cleaned dataset saved to '{new_fd}'")
        print(f"all_pairs.csv and all_smiles.csv saved to '{utils_dir}'")
    except Exception as e:
        print(f"Error processing files in '{folder_path}': {e}")

if __name__ == "__main__":
    # Check if the user provided a folder path as an argument
    if len(sys.argv) < 3:
        print("Usage: python script_name.py <folder_path> <utils_path>")
        sys.exit(1)

    # Get the folder path and utils path from the command-line arguments
    folder_path = sys.argv[1]
    utils_path = sys.argv[2]

    # Call the processing function
    process_smiles_data(folder_path, utils_path)
