# Initially, I was going to include an RNN in the comparison, but it performed extremely poorly compared to both the LSTM and CfC and just served no real purpose

This directory includes all files needed for the evaluation of the CfC and LSTM on the PJME_Hourly electricity prices dataset

rnn.py: 3-layer RNN implementation  
lstm.py: 3-layer LSTM implementation  
dataprep.py: Prepare electricity dataset, add relevant features (for example, holiday days where electricity consumption is usually higher)  
showlosses.py: Generate figure 6  
traincfc.py: Implement and train CfC (CfC implementation is open source via https://github.com/mlech26l/ncps)  
trainlstm.py: Train LSTM model  
trainrnn.py: Train RNN model
