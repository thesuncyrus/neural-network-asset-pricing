"""Execute the supervised autoencoder notebook and export aggregate results."""
from pathlib import Path
import argparse,json,os
parser=argparse.ArgumentParser()
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,default=Path('artifacts/notebook'))
args=parser.parse_args()
os.environ['EQUITY_DATA_DIR']=str(args.data_dir.resolve())
out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
notebook=json.loads((Path(__file__).parent/'notebooks/supervised_autoencoder.ipynb').read_text())
os.chdir(out)
context={'__name__':'__main__'}
for i,cell in enumerate(notebook['cells']):
    if cell['cell_type']=='code':
        print(f'Executing notebook cell {i}',flush=True)
        exec(compile(''.join(cell['source']),f'notebook-cell-{i}','exec'),context)
context['metrics_df'].to_csv('metrics-by-year.csv',index=False)
context['summary'].to_csv('model-comparison.csv')
context['ls_summary'].to_csv('decile-portfolios.csv',index=False)
for model,path in context['ls_paths'].items():
    path.to_csv(f'decile-path-{model}.csv',index=False)
print('Notebook workflow completed.')
