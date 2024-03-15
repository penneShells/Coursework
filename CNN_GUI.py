import  PySimpleGUI as sg
import os
import DataCollect_CV2
import CNN_Train
import modelTesting


#The generate data button will take each video in the videos folder and extract a face from each frame of video, saving
#it to the datasets folder. There is a premade dataset in the folder already.
#If you do decide to test the data generation, probably don't use it for testing as it does generate some eerroneous data
#which I remove manually. Just use the premade dataset.

#The training process is HIGHLY RESOURCE INTENSIVE and takes a long time, also depending on the configuration of the
#network and hyperparameters. It will generate a keras file which can be


class GUI:
    def __init__(self):
        #Get functions from other files
        self.dataGenerate = DataCollect_CV2.collectAll
        self.train = CNN_Train.train
        self.running = True
        self.test = modelTesting.testAll

        #Declare arrays for possible hyper parameter options for dropdowns
        datasets = [name.upper() for name in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", name))]
        batchSizes = [16, 32, 48, 64, 128, 192, 256]
        imageSizes = [16, 32, 64, 128, 256, 384, 512]
        epochnum = [1, 5, 10, 15, 20, 25, 50]
        dropouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        #To be appended to a filename to create a new keras file without overwrite complications
        self.suffix = len([model for model in os.listdir("keras_models")]) + 1

        #GUI structuring
        self.column_0 = [
            [sg.Button("Generate Training Data", size=(30, 10))]
        ]

        self.column_1 = [
            [sg.Text("Training configuration", size=(30, 1), justification="center")],
            [sg.DropDown(datasets, key='-DATADROP-', default_value="DATASET", size=(30, 1))],
            [sg.DropDown(batchSizes, key="-BATCHDROP-", default_value="BATCH SIZE", size=(30, 1))],
            [sg.DropDown(imageSizes, key='-SIZEDROP-', default_value="IMAGE SIZE", size=(30, 1))],
            [sg.DropDown(epochnum, key='-EPOCHDROP-', default_value="EPOCHS", size=(30, 1))],
            [sg.DropDown(dropouts, key='-DROPOUTDROP-', default_value="DROPOUT", size=(30, 1))]
        ]
        #hmm

        self.column_2 = [
            [sg.Button("Begin Training", size=(30, 10))],
        ]

        self.layout = [
            [
                sg.Column(self.column_0),
                sg.VSeperator(),
                sg.Column(self.column_1),
                sg.VSeperator(),
                sg.Column(self.column_2),
            ]
        ]

        self.window = sg.Window("OpenCV Integration", self.layout)

    #Closes the window and begins the model training process
    def trainThenQuit(self, values, suffix):
        self.running = False
        gui.window.close()
        model = self.train(values, suffix)
        self.test(model, values[2])


    #Executes functions based on button events
    #Prevents the need for many if statements, because I would never use those.
    def eventChecking(self, event, values):
        possibleEvents = {
            "Generate Training Data": (self.dataGenerate, None, None),
            "Begin Training": (self.trainThenQuit, values, self.suffix),
        }

        eventResponse = possibleEvents.get(event)

        #If statement
        if eventResponse is not None:
            eventResponse[0](eventResponse[1], eventResponse[2])


if __name__ == "__main__":
    #Instantiate the GUI and loop to detect button events
    gui = GUI()

    while gui.running:
        event, values = gui.window.read(timeout=20)
        gui.eventChecking(event, values)
