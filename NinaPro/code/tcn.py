class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, N_C, Ch_in, Ch_out, KS, stride, dilation, dropout_p=0.2):
        super(TemporalBlock, self).__init__()
               
        module=[]
        for n in range(N_C):
           
            if n==0:
                input_channels=Ch_in
               
            else:
                input_channels=Ch_out
           
           
            conv = weight_norm(nn.Conv1d(input_channels, Ch_out, KS[n], stride=stride[n], padding=(KS[n]-1)*dilation[n],\
                                  dilation=dilation[n]))
           
            conv.weight.data.normal_(0, 0.01)

            ## spike
           
            chomp = Chomp1d((KS[n]-1)*dilation[n])
           
            relu = nn.ReLU()
            dropout= nn.Dropout(dropout_p)
           
            module.append(conv)
           
            if stride[n]==1:
                module.append(chomp)
               
            module.append(relu)
            module.append(dropout)
       
       
       
        self.net = nn.Sequential(*module)
       
        self.downsample = nn.Conv1d(Ch_in, Ch_out, 1) if Ch_in != Ch_out else None
        self.relu = nn.ReLU()
       


    def forward(self, x):
       
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
       
        return self.relu(out + res)



class TCN(nn.Module):
    def __init__(self, N_C, input_size, output_size, num_channels, K_S, Dil, Str, dropout, F_Ns, dropout_F, \
                 Receptive_F):
       
        super(TCN, self).__init__()
       
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
           
            dilation_size = Dil[i]
            stride_size=Str[i]
           
           
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
           
           
            layers += [TemporalBlock(N_C, in_channels, out_channels, K_S[i], stride=Str[i], dilation=Dil[i],
                                      dropout_p=dropout)]

        self.network = nn.Sequential(*layers).to(device)
       
       
        TCN_out=num_channels[-1]
       
        module_F=[]
       
        self.F_Ns=F_Ns
        module_F.append(nn.Linear(TCN_out,F_Ns[-1]))
               
        self.F=nn.Sequential(*module_F).to(device)
       
        self.Receptive_F=Receptive_F
           

    def forward(self, inputs):
       
        """Inputs have to have dimension (N, C_in, L_in)"""
       
        batch_size=inputs.size()[0]
       
        y_temp = self.network(inputs)  # input should have dimension (N, C, L)
       
        y_temp=y_temp[:,:,self.Receptive_F:].clone()
       
        y_temp=y_temp.transpose(1,2)
       
        out=self.F(y_temp.reshape([-1,y_temp.size()[-1]]))
       
        out=out.reshape([batch_size,y_temp.size()[1],-1]).transpose(1,2)
       
        return out

# Dil=[ [1,1], [2,2], [4,4], [8,8], [8,8], [8,8],  [16,16] ]
# Dil=[ [1,1], [2,2], [4,4], [8,8], [16,16], [16,16],  [32,32] ]
# #K_S=[ [7,7], [7,7], [7,7], [7,7], [7,7], [7,7], [7,7] ]
# K_S=[ [4,4], [4,4], [4,4], [4,4], [4,4], [4,4], [4,4] ]
# Str=[ [1,1] ]*7

# print( np.sum((np.array(K_S)-1)*np.array(Dil)) )
# Receptive_F=np.sum((np.array(K_S)-1)*np.array(Dil))

# F_Ns=torch.tensor([32,5])
# dropout_F=0.1
# dropout_cnn=0.2

# tcn=TCN(N_C, input_size, output_size, num_channels, K_S, Dil, Str, \
#         dropout=dropout_cnn, F_Ns=F_Ns, dropout_F=dropout_F,Receptive_F=Receptive_F )