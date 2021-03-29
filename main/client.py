import os
import torch
import pickle
import threading
import torch.nn as nn

from .Done import Done
from .FLPacket import FLPacket, CONST_STATUS, CONST_TYPE
from .LocalVNetwork import STCPSocket
from .LocalVNetwork import StandardPrint
from .LocalVNetwork import LocalNode, ForwardNode
from .NeuronNetwork import SimpleDataset, train, test

MODEL_FILE_NAME = "model.weight"


class Client(object):
    def __init__(self, server_address, architecture, data, directory, verbosities={"user": ["notification"]}) -> None:
        self.__socket__ = STCPSocket()
        self.__print__ = StandardPrint("Client", verbosities)
        
        self.__server_address__ = server_address
        
        self.__architecture__ = architecture
        self.__data__, self.__label__ = data[0], data[1]
        
        self.__directory__ = directory
        self.__model_path__ = os.path.join(self.__directory__, MODEL_FILE_NAME)
        if not os.path.isdir(self.__directory__):
            os.mkdir(self.__directory__)
        
        self.__node__ = LocalNode()
        self.__forwarder__ = ForwardNode(self.__node__, self.__socket__)
        
    def __require__(self):
        try:
            packet = FLPacket(CONST_TYPE.REQUIRE, CONST_STATUS.NONE)
            self.__node__.send(self.__forwarder__.name, packet.create())

            _, data, _ = self.__node__.recv(self.__forwarder__.name)
            packet_dict = FLPacket.extract(data)
            result = FLPacket.check(packet_dict, CONST_TYPE.REQUIRE, CONST_STATUS.ACCEPT, is_dict=True)
            if result.value == False:
                return result

            model = self.__architecture__()
            model.deserialize(packet_dict["DATA"])
            model.write(self.__model_path__)
            self.__print__("dev", "debug", "Test when receiving model")
            test(model, (3, 2))
            return Done(True)
        except Exception as e:
            return Done(False, {"user": {"error": "Something wrong in requiring model"}, "dev": {"error": repr(e)}})
        
    def __train__(self):
        try:
            dataset = SimpleDataset(self.__data__, self.__label__)
            nepochs = 1000
            batchsize = 8
            model = self.__architecture__()
            model.read(self.__model_path__)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr = 5e-4)
            train(model, dataset, loss_fn, optimizer, nepochs, batchsize, brief_print= True)
            model.write(self.__model_path__)
            self.__print__("dev", "debug", "Test after training model")
            test(model, (3, 2))
            return Done(True)
        except Exception as e:
            return Done(False, {"user": {"notification": "Something wrong in training"}, "dev": {"error": repr(e)}})

    def __submit__(self):
        try:
            packet = FLPacket(CONST_TYPE.SUBMIT, CONST_STATUS.NONE)
            self.__node__.send(self.__forwarder__.name, packet.create())

            _, data, _ = self.__node__.recv(self.__forwarder__.name)
            result = FLPacket.check(data, CONST_TYPE.SUBMIT, CONST_STATUS.ACCEPT)
            if result.value == False:
                return result

            packet = FLPacket(CONST_TYPE.SUBMIT, CONST_STATUS.UPLOAD)
            model = self.__architecture__()
            model.read(self.__model_path__)
            self.__print__("dev", "debug", "Test when sending model")
            test(model, (3, 2))

            obj = pickle.dumps((self.__directory__, model.serialize()))
            packet.set_data(obj)
            
            self.__node__.send(self.__forwarder__.name, packet.create())

            _, data, _ = self.__node__.recv(self.__forwarder__.name)
            result = FLPacket.check(data, CONST_TYPE.SUBMIT, CONST_STATUS.SUCCESS)
            if result.value == False:
                return result

            return Done(True)
        except Exception as e:
            return Done(False, {"user": {"notification": "Something wrong in submiting"}, "dev": {"error": repr(e)}})

    def start(self):
        self.__socket__.connect(self.__server_address__)
        forwarder_thread = threading.Thread(target= self.__forwarder__.start)
        forwarder_thread.start()

        result = self.__require__()
        self.__print__.use_dict(result.print_dict)
        result = self.__train__()
        self.__print__.use_dict(result.print_dict)
        result = self.__submit__()
        self.__print__.use_dict(result.print_dict)
        self.__socket__.close()
