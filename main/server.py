import os
import time
import errno
import pickle
import threading

from .LocalVNetwork import StandardPrint
from .LocalVNetwork import STCPSocket, SecureTCP
from .LocalVNetwork import LocalNode, ForwardNode
from .NeuronNetwork import test

from .FLPacket import FLPacket, CONST_TYPE, CONST_STATUS

MODEL_FILE_NAME = "SimpleNeuronNetwork01.weight"
MIN_CLIENTS_TO_UPDATE = 2

class ClientList(object):
    def __init__(self, path):
        self.__path__ = path
        self.__list__ = []
        self.__lock__ = threading.Lock()
        if not os.path.isfile(self.__path__):
            self.__write__()
        self.__read__()

    def __read__(self):
        f = open(self.__path__, "rb")
        self.__list__ = pickle.load(f)
        f.close()

    def __write__(self):
        f = open(self.__path__, "wb")
        pickle.dump(self.__list__, f)
        f.close()

    def add(self, client_name):
        self.__lock__.acquire()
        self.__read__()
        if client_name not in self.__list__:
            self.__list__.append(client_name)
        self.__write__()
        self.__lock__.release()

    def get(self):
        self.__lock__.acquire()
        self.__read__()
        self.__lock__.release()
        return self.__list__.copy()
        

    def clear(self):
        self.__lock__.acquire()
        self.__list__ = []
        self.__write__()
        self.__lock__.release()

class Responser(object):
    def __init__(self, architecture, directory, clients_list, client_socket: STCPSocket, client_address, verbosities={"user": ["notification"]}) -> None:
        self.__socket__ = client_socket
        self.__print__ = StandardPrint("Reponser of {}".format(client_address), verbosities)
        
        self.__architecture__ = architecture
        
        self.__directory__ = directory
        self.__model_path__ = os.path.join(self.__directory__, MODEL_FILE_NAME)
        
        self.__clients_list__ = clients_list

        self.__node__ = LocalNode()
        self.__forwarder__ = ForwardNode(self.__node__, self.__socket__)
        forwarder_thread = threading.Thread(target= self.__forwarder__.start)
        forwarder_thread.start()

    def start(self):
        while True:
            try:
                source, data, _ = self.__node__.recv()
                if source == None:
                    self.__socket__.sendall(b"$test alive")
                    # if connection is alive, ignore error, print a warning and continue
                    self.__print__("dev", "warning", "Something error when socket is None but it still connects")
                    continue
            except Exception as e:
                if e.args[0] in (errno.ENOTSOCK, errno.ECONNREFUSED, errno.ECONNRESET, errno.EBADF):
                    self.__print__("dev", "warning", "Connection closed")
                else:
                    self.__print__("dev", "error", "Some error occurs when receving data from LocalNode")
                break

            packet_dict = FLPacket.extract(data)

            if source == self.__forwarder__.name and packet_dict["TYPE"] == CONST_TYPE.REQUIRE:
                packet = FLPacket(CONST_TYPE.REQUIRE, CONST_STATUS.ACCEPT)
                model = self.__architecture__()
                model.read(self.__model_path__)
                packet.set_data(model.serialize())
                self.__node__.send(self.__forwarder__.name, packet.create())
                self.__print__("dev", "debug", "Test after sending model")
                test(model, (3, 2))
            elif source == self.__forwarder__.name and packet_dict["TYPE"] == CONST_TYPE.SUBMIT:
                try:
                    packet = FLPacket(CONST_TYPE.SUBMIT, CONST_STATUS.ACCEPT)
                    self.__node__.send(self.__forwarder__.name, packet.create())
                    
                    _, data, _ = self.__node__.recv(self.__forwarder__.name)
                    packet_dict = FLPacket.extract(data)
                    result = FLPacket.check(packet_dict, CONST_TYPE.SUBMIT, CONST_STATUS.UPLOAD, is_dict= True)
                    if result.value == False:
                        raise Exception(result.print_dict["user"]["warning"])
                    
                    client_name, model_obj = pickle.loads(packet_dict["DATA"])

                    client_path = os.path.join(self.__directory__, client_name)
                    if not os.path.isdir(client_path):
                        os.mkdir(client_path)

                    model = self.__architecture__()
                    model.deserialize(model_obj)
                    model.write(os.path.join(client_path, "model.weight"))
                    self.__print__("dev", "debug", "Test when receiving model")
                    test(model, (3, 2))

                    packet = FLPacket(CONST_TYPE.SUBMIT, CONST_STATUS.SUCCESS)
                    self.__node__.send(self.__forwarder__.name, packet.create())

                    self.__clients_list__.add(client_name)
                except Exception as e:                 
                    packet = FLPacket(CONST_TYPE.SUBMIT, CONST_STATUS.DENY)
                    self.__node__.send(self.__forwarder__.name, packet.create())
                    self.__print__("dev", "error", repr(e))


        self.__print__("user", "notification", "Client leaves")
        self.__forwarder__.close()
        self.__node__.close()


class Listener(object):
    def __init__(self, server_address, architecture, directory, verbosities={"user": ["notification"]}) -> None:
        self.__socket__ = STCPSocket()
        self.__print__ = StandardPrint("Listener", verbosities)
        
        self.__server_address__ = server_address
        
        self.__architecture__ = architecture

        self.__directory__ = directory
        if not os.path.isdir(self.__directory__):
            os.mkdir(self.__directory__)

        self.__model_path__ = os.path.join(self.__directory__, MODEL_FILE_NAME)

        self.__clients_list__ = ClientList(os.path.join(self.__directory__, "clients.list"))
        
    def __averaging_model__(self):
        self.__print__("user", "notification", "Begin averaging model...")
        clients = self.__clients_list__.get()
        client_models = []
        for client in clients:
            client_path = os.path.join(self.__directory__, client)
            client_model_path = os.path.join(client_path, "model.weight")
            client_model = self.__architecture__()
            client_model.read(client_model_path)
            if client_model.checkVersion(self.__model_path__) == True:
                client_models.append(client_model)

        if len(client_models) < MIN_CLIENTS_TO_UPDATE:
            self.__print__("user", "notification", "Averaging model fails (not enough number of client models)")
            return

        server_model = self.__architecture__()
        server_model.read(self.__model_path__)
        server_model.averaging(client_models)
        server_model.nextVersion()
        server_model.write(self.__model_path__)
        self.__print__("dev", "debug", "Test after averaging")
        test(server_model, (3, 2))
        self.__clients_list__.clear()
        self.__print__("user", "notification", "Averaging model successes")

    def start(self) -> None:        
        self.__socket__.bind(self.__server_address__)
        self.__socket__.listen()
        self.__print__("user", "notification", "Server is listening...")
        
        averaging_thread = threading.Thread(target= self.averaging_periodically)
        averaging_thread.start()

        while True:
            client_socket, client_address = self.__socket__.accept()
            self.__print__("user", "notification",
                           "Client from {} connected".format(client_address))
            responser = Responser(
                architecture= self.__architecture__, 
                directory= self.__directory__, 
                clients_list= self.__clients_list__,
                client_socket= client_socket, 
                client_address= client_address, 
                verbosities= {
                        "user": ["notification"], 
                        "dev": ["debug"]
                    }
                )
            responser_thead = threading.Thread(target=responser.start)
            responser_thead.start()
    
    def averaging_periodically(self):
        while True:
            # Update after every three minutes
            time.sleep(180)
            self.__averaging_model__()