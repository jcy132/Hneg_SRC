from trainer import Trainer
import os
if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
    trainer = Trainer('supernet')
    trainer.start()
    print('Supernet training finished!!!')
