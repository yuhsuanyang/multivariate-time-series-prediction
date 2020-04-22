import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# 1dcnn attention with GRU seq2seq network 

class Encoder(nn.Module):
    def __init__(self,input_dim,cnn_hidden_size,cnn_kernel_size): 
        super(Encoder,self).__init__()
        self.input_dim=input_dim
        self.cnn_hidden_size=cnn_hidden_size
        self.cnn_kernel_size=cnn_kernel_size
        self.pooling=nn.AvgPool1d(kernel_size=2)
        #self.pooling=nn.MaxPool1d(kernel_size=2)
        self.convs=nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=cnn_hidden_size,kernel_size=cnn_kernel_size) for i in range(input_dim)])
        
    
    def forward(self,x):
        main_vec=F.relu(self.convs[0](x[:,:,0].unsqueeze(1)))
        main_vec=self.pooling(main_vec).flatten(start_dim=1).unsqueeze(1) #shape:(batch, 1, output_shape)
        if self.input_dim==1:
            return main_vec
        output_shape=main_vec.shape[2]
        #print(main_vec.shape)
        attn_weights=Variable(x.data.new(x.size(0), 1,self.input_dim-1).zero_())
        aux_vec=Variable(x.data.new(x.size(0), self.input_dim-1,output_shape).zero_())
       
        for i in range(1,self.input_dim):
            others=F.relu(self.convs[i](x[:,:,i].unsqueeze(1)))
            others=self.pooling(others).flatten(start_dim=1)
           
            aux_vec[:,i-1,:]=others
            attn_weights[:,:,i-1]=main_vec.bmm(others.unsqueeze(2)).squeeze(2) #dot attention
        
        attn_weights=F.softmax(attn_weights,dim=2)
        
        weighted_aux_vec=attn_weights.bmm(aux_vec)
        
        return main_vec, torch.cat([main_vec,weighted_aux_vec],dim=2), attn_weights
    
    def count_vec_size(self,input_len):
        conv_output_size=(input_len-(self.cnn_kernel_size-1)-1)+1
        pooling_output_size=int(conv_output_size/2)
        return pooling_output_size*self.cnn_hidden_size
    
    
class Decoder(nn.Module):
    def __init__(self,hidden_size,output_length,fc_size):
        super(Decoder,self).__init__()
        self.output_length=output_length
        
        self.rnn=nn.GRU(1,hidden_size,batch_first=True)
        self.fc1=nn.Linear(hidden_size,fc_size)
        self.fc2=nn.Linear(fc_size,1) #最終輸出是一個值
        
        
        
    def forward(self,xn,context_vec):
        decoder_input=xn #xn 是input seq的最後一點
        
        decoder_hidden=context_vec #First decoder hidden state will be last encoder hidden state
        result=Variable(xn.data.new(xn.size(0),self.output_length,1).zero_())
       
                    
        for i in range(self.output_length):
            decoder_output,decoder_hidden=self.rnn(decoder_input,decoder_hidden)
            #print('decoder output')
            #print(decoder_output.shape)
            decoder_output=self.fc1(decoder_output) 
            decoder_output=F.relu(decoder_output)
            decoder_input=self.fc2(decoder_output)
            #print('decoder input')
            #print(decoder_input.shape)
            result[:,i,:]=decoder_input.squeeze(2)
        return result    

class S2S_cnn_attn(nn.Module):
    def __init__(self,cnn_parameters,fc_size,input_dim,input_length,output_length):
        #cnn_parameters=(cnn_hidden_size,kernel_size)
        super(S2S_cnn_attn,self).__init__()
        
        self.input_dim=input_dim
        self.output_length=output_length
        
        self.encoder=Encoder(input_dim,cnn_parameters[0],cnn_parameters[1])
        if input_dim==1:
            rnn_hidden_size=self.encoder.count_vec_size(input_length)
        else:
            rnn_hidden_size=2*self.encoder.count_vec_size(input_length)
        self.decoder=Decoder(rnn_hidden_size,output_length,fc_size)
       
    
    def forward(self,x):
        _,context_vec, self.attn_weights=self.encoder(x)
        context_vec=context_vec.permute(1,0,2)
        
        #print(context_vec.shape)
        the_last_point=x[:,-1,:1].unsqueeze(1)
        #print('the last point')
        #print(the_last_point.shape)
        output=self.decoder(the_last_point,context_vec)
        return output
               