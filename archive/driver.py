from datasets import load_dataset
from src.datagraph import DataGraph # import the DataGraph class
from src.gnn import GNN # import the GNN class

def main():
	# download the imdb sentiment analysis dataset from the internet
	print("[INFO] downloading dataset...")
	dataset = load_dataset("imdb")

	# create graph structures from the dataset
	print("[INFO] creating graph structures...")
	train_graphs = []
	for i in range(len(dataset["train"]["text"][:300])): # iterate over the dataset
		graph = DataGraph(dataset["train"]["text"][i], dataset["train"]["label"][i]) # create a DataGraph object from the current text
		train_graphs.append(graph) # add the graph to the list of graphs

	test_graphs = []
	for i in range(len(dataset['test']['text'][:100])):
		graph = DataGraph(dataset['test']['label'][i])
		test_graphs.append(graph)
	
	


if __name__ == "__main__":
	main()