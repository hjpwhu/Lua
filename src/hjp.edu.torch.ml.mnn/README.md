## End-to-End Memory Networks in TensorFlow for Language Model

End-to-End Memory Networks implementation in [Torch](http://torch.ch/) for language model on the Penn Treebank (ptb) data.


### Dependencies

You will need the following packages:
* nn
* nngraph
* paths
* xlua


### Quickstart

The ptb data is from:

* [Wojciech Zaremba's lstm repo](https://github.com/wojzaremba/lstm)

and put the 'data' folder into current directionary.
	
Use 'main.lua' for running a model with 2 hops and memory size of 20, run the following command:

	$ th main.lua --nhop 2 --memsize 20
	
To see all training options, run the following command:

	$ th main.lua --help
	
	
### Reference

The memory model is from:
* [Memory Networks](https://arxiv.org/pdf/1410.3916.pdf). Weston et al., ICLR 2016.
* [End-to-End Memory Networks](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf). Sukhbaatar et al., NIPS 2015.


### Acknowledgments

Our implementation utilizes code from the following:
* [facebook's MemNN repo](https://github.com/facebook/MemNN)


### License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

