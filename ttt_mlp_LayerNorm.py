import torch
from torch import nn, Tensor




     
class MLP(nn.Module):

    def __init__(self,dim):
        super().__init__()
        self.proj_1 =  nn.Linear(dim,dim,bias=False)
        self.proj_2 =  nn.Linear(dim,dim,bias=False)        
        self.gelu = nn.GELU()
       
             	   
    def forward(self, x):
       
        x = self.proj_1(x)
        x = self.gelu(x)          
        x = self.proj_2(x)
        
                          
        return x
        

class LocalMappingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
        self.pre_norm = nn.LayerNorm(dim,elementwise_affine=False) 
      
        self.mapping = MLP(dim)
      
             	   
    def forward(self, x):
    
        x = self.pre_norm(x)      
        x = self.mapping(x)    
      

        return x



   
    

class TTT(nn.Module):
   

    def __init__(self, dim: int):
        super(TTT, self).__init__()
       
     

       
        self.mapping = MLP(dim)
        self.state =  nn.Linear(dim,dim,bias=False)
        self.probe =  nn.Linear(dim,dim,bias=False)
        
       
    def forward(self, in_seq: Tensor) -> Tensor:

       
        outs = []
        
        for seq in range(in_seq.size(1)):
            
            state = self.state(in_seq[:,seq,:])
            train_view = state + torch.randn_like(state)
            label_view = state
            loss = nn.functional.mse_loss(self.mapping(train_view), label_view)
            grads = torch.autograd.grad(
                loss, self.mapping.parameters(),create_graph=True)
            with torch.no_grad():
                for param, grad in zip(self.mapping.parameters(), grads):
              
                    param -= 0.01 * grad
            readout = self.mapping(self.probe(in_seq[:,seq,:])).detach()
            outs.append(readout)
        out = torch.stack(outs, dim=1)
        
        return out
        


    	

class GlobalMappingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
             
        self.pre_norm = nn.LayerNorm(dim,elementwise_affine=False) 
        
        self.ttt = TTT(dim)       
        
              
                                      	   
    def forward(self, x):
    
        x = self.pre_norm(x)       
        x = self.ttt(x)
     

        return x         




class TTTBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
       
         
        self.local_mapping = LocalMappingUnit(d_model)
        self.global_mapping = GlobalMappingUnit(d_model)
        
    
        
        
        
    def forward(self, x):
                  
        residual = x
        
        x = self.global_mapping(x)
    
        x = x + residual
        
        residual = x
        
        x = self.local_mapping(x)
        
                                          
        out = x + residual
        
        
        return out



class TTTM(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[TTTBlock(d_model) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)















