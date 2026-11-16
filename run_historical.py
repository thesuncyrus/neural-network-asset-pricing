"""Execute the cleaned historical notebook locally, keeping artifacts private."""
from pathlib import Path
import argparse,json,os
parser=argparse.ArgumentParser()
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,default=Path('artifacts/historical'))
args=parser.parse_args()
os.environ['EQUITY_DATA_DIR']=str(args.data_dir.resolve())
out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
notebook=json.loads((Path(__file__).parent/'notebooks/historical_supervised_autoencoder.ipynb').read_text())
os.chdir(out)
context={'__name__':'__main__'}
for i,cell in enumerate(notebook['cells']):
    if cell['cell_type']=='code':
        print(f'Executing historical cell {i}',flush=True)
        exec(compile(''.join(cell['source']),f'historical-cell-{i}','exec'),context)
context['metrics_df'].to_csv('rerun-metrics-by-year.csv',index=False)
context['summary'].to_csv('rerun-model-comparison.csv')
context['ls_summary'].to_csv('rerun-decile-portfolios.csv',index=False)
for model,path in context['ls_paths'].items():
    path.to_csv(f'rerun-decile-path-{model}.csv',index=False)
print('Historical workflow completed. Outputs are newly computed, not the saved historical tables.')
