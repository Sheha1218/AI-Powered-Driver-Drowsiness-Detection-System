import torch
from pipeline import neuralnet



class detect:
    def __init__(self,model_path, class_names):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = neuralnet().to(self.device)
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.model.eval()
        
        self.class_names = class_names
    
    def predict(self,input_tensor):
        input_tensor = input_tensor.to(self.device)
    
        with torch.no_grad():
            output =self.model(input_tensor)
            probs = torch.softmax(output,dim=1)
            class_id =torch.argmax(probs,dim=1).item()
            confidence = probs[0,class_id].item()
            
        return self.class_names[class_id], confidence