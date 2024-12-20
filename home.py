import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
import threading
import shutil
from facerec import *
from register import *
from face_detection import *
from handler import *
import time
import csv
import numpy as np
import ntpath
import os


active_page = 0
thread_event = None
left_frame = None
right_frame = None
heading = None
webcam = None
img_label = None
img_read = None
img_list = []
slide_caption = None
slide_control_panel = None
current_slide = -1

root = tk.Tk()
root.geometry("1000x900+200+100")

# create Pages
pages = []
for i in range(7):
    pages.append(tk.Frame(root, bg="#e1e8df"))
    pages[i].pack(side="top", fill="both", expand=True)
    pages[i].place(x=0, y=0, relwidth=1, relheight=1)


def goBack():
    global active_page, thread_event, webcam
    
    if (active_page==6 and not thread_event.is_set()):
        thread_event.set()
        webcam.release()

    for widget in pages[active_page].winfo_children():
        widget.destroy()

    pages[0].lift()
    active_page = 0


def basicPageSetup(pageNo):
    global left_frame, right_frame, heading

    back_img = tk.PhotoImage(file= r"E:\FinalYearProjectDetails\MajorprojectFacialRecognition\Facial-Recognition-for-Crime-Detection\img\back.png")
    back_button = tk.Button(pages[pageNo], image=back_img, bg="#e1e8df", bd=0, highlightthickness=0,
           activebackground="#e1e8df", command=goBack)
    back_button.image = back_img
    back_button.place(x=2, y=2)

    heading = tk.Label(pages[pageNo], fg="black", bg="#e1e8df", font="Arial 20 bold", pady=10)
    heading.pack()

    content = tk.Frame(pages[pageNo], bg="#e1e8df", pady=20)
    content.pack(expand="true", fill="both")

    left_frame = tk.Frame(content, bg="#e1e8df")
    left_frame.grid(row=0, column=0, sticky="nsew")

    right_frame = tk.LabelFrame(content, text="Detected Criminals", bg="#e1e8df", font="Arial 20 bold", bd=4,
                             foreground="#2ea3ef", labelanchor="n")
    right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    content.grid_columnconfigure(0, weight=1, uniform="group1")
    content.grid_columnconfigure(1, weight=1, uniform="group1")
    content.grid_rowconfigure(0, weight=1)

 # Add logo to the left frame at the bottom-left corner
    logo = tk.PhotoImage(file=r"E:\FinalYearProjectDetails\MajorprojectFacialRecognition\Facial-Recognition-for-Crime-Detection\colours.png")
    logo_label = tk.Label(left_frame, image=logo, bg="#e1e8df", bd=0)
    logo_label.pack(side='bottom', padx=10, pady=10, anchor='sw')  # Adjust padx and pady values as needed

def showImage(frame, img_size):
    global img_label, left_frame

    img = cv2.resize(frame, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    if (img_label == None):
        img_label = tk.Label(left_frame, image=img, bg="#202d42")
        img_label.image = img
        img_label.pack(padx=20)
    else:
        img_label.configure(image=img)
        img_label.image = img


def getNewSlide(control):
    global img_list, current_slide

    if(len(img_list) > 1):
        if(control == "prev"):
            current_slide = (current_slide-1) % len(img_list)
        else:
            current_slide = (current_slide+1) % len(img_list)

        img_size = left_frame.winfo_height() - 200
        showImage(img_list[current_slide], img_size)

        slide_caption.configure(text = "Image {} of {}".format(current_slide+1, len(img_list)))


def selectMultiImage(opt_menu, menu_var):
    global img_list, current_slide, slide_caption, slide_control_panel

    filetype = [("images", "*.jpg *.jpeg *.png")]
    path_list = filedialog.askopenfilenames(title="Choose atleast 5 images", filetypes=filetype)

    if(len(path_list) < 5):
        messagebox.showerror("Error", "Choose atleast 5 images.")
    else:
        img_list = []
        current_slide = -1

        # Resetting slide control panel
        if (slide_control_panel != None):
            slide_control_panel.destroy()

        # Creating Image list
        for path in path_list:
            img_list.append(cv2.imread(path))

        # Creating choices for profile pic menu
        menu_var.set("")
        opt_menu['menu'].delete(0, 'end')

        for i in range(len(img_list)):
            ch = "Image " + str(i+1)
            opt_menu['menu'].add_command(label=ch, command= tk._setit(menu_var, ch))
            menu_var.set("Image 1")


        # Creating slideshow of images
        img_size =  left_frame.winfo_height() - 200
        current_slide += 1
        showImage(img_list[current_slide], img_size)

        slide_control_panel = tk.Frame(left_frame, bg="#202d42", pady=20)
        slide_control_panel.pack()

        back_img = tk.PhotoImage(file="E:\FinalYearProjectDetails\MajorprojectFacialRecognition\Facial-Recognition-for-Crime-Detection\img\previous.png")
        next_img = tk.PhotoImage(file="E:\FinalYearProjectDetails\MajorprojectFacialRecognition\Facial-Recognition-for-Crime-Detection\img\previous.png")

        prev_slide = tk.Button(slide_control_panel, image=back_img, bg="#202d42", bd=0, highlightthickness=0,
                            activebackground="#202d42", command=lambda : getNewSlide("prev"))
        prev_slide.image = back_img
        prev_slide.grid(row=0, column=0, padx=60)

        slide_caption = tk.Label(slide_control_panel, text="Image 1 of {}".format(len(img_list)), fg="#ff9800",
                              bg="#202d42", font="Arial 15 bold")
        slide_caption.grid(row=0, column=1)

        next_slide = tk.Button(slide_control_panel, image=next_img, bg="#202d42", bd=0, highlightthickness=0,
                            activebackground="#202d42", command=lambda : getNewSlide("next"))
        next_slide.image = next_img
        next_slide.grid(row=0, column=2, padx=60)


def register(entries, required, menu_var):
    global img_list

    # Checking if no image selected
    if(len(img_list) == 0):
        messagebox.showerror("Error", "Select Images first.")
        return

    # Fetching data from entries
    entry_data = {}
    for i, entry in enumerate(entries):
        # print(i)
        val = entry[1].get()
        # print(val)

        if (len(val) == 0 and required[i] == 1):
            messagebox.showerror("Field Error", "Required field missing :\n\n%s" % (entry[0]))
            return
        else:
            entry_data[entry[0]] = val.lower()


    # Setting Directory
    path = os.path.join('face_samples', "temp_criminal")
    if not os.path.isdir(path):
        os.mkdir(path)

    no_face = []
    for i, img in enumerate(img_list):
        # Storing Images in directory
        id = registerCriminal(img, path, i + 1)
        if(id != None):
            no_face.append(id)

    # check if any image doesn't contain face
    # if(len(no_face) > 0):
    #     no_face_st = ""
    #     for i in no_face:
    #         no_face_st += "Image " + str(i) + ", "
    #     messagebox.showerror("Registration Error", "Registration failed!\n\nFollowing images doesn't contain"
    #                     " face or Face is too small:\n\n%s"%(no_face_st))
    #     shutil.rmtree(path, ignore_errors=True)
    
    else:
        # Storing data in database
        insertData(entry_data)
        rowId = 1
        if rowId >= 0:
            messagebox.showinfo("Success", "Person Registered Successfully.")
            shutil.move(path, os.path.join('face_samples', entry_data["Name"]))

            # Save profile pic with filename 
            profile_img_num = int(menu_var.get().split(' ')[1]) - 1
            if not os.path.isdir("profile_pics"):
                os.mkdir("profile_pics")

            filename = entry_data["Name"] + ".png"

            profile_pic_filename = os.path.join("profile_pics", filename)
            cv2.imwrite(profile_pic_filename, img_list[profile_img_num])

            goBack()
        else:
            shutil.rmtree(path, ignore_errors=True)
            messagebox.showerror("Database Error", "Some error occured while storing data.")


## update scrollregion when all widgets are in canvas
def on_configure(event, canvas, win):
    canvas.configure(scrollregion=canvas.bbox('all'))
    canvas.itemconfig(win, width=event.width)

## Register Page ##
def getPage1():
    global active_page, left_frame, right_frame, heading, img_label
    active_page = 1
    img_label = None
    opt_menu = None
    menu_var = tk.StringVar(root)
    pages[1].lift()

    basicPageSetup(1)
    heading.configure(text="Register a Person", fg="black", bg="#e1e8df")
    right_frame.configure(text="Enter Details", fg="black", bg="#e1e8df")

    btn_grid = tk.Frame(left_frame, bg="#e1e8df")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Images", command=lambda: selectMultiImage(opt_menu, menu_var), font="Arial 15 bold", bg="#000000",
           fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#e1e8df",
           activeforeground="white").grid(row=0, column=0, padx=25, pady=25)


    # Creating Scrollable Frame
    canvas = tk.Canvas(right_frame, bg="#e1e8df", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand="true", padx=30)
    scrollbar = tk.Scrollbar(right_frame, command=canvas.yview, width=20, troughcolor="#e1e8df", bd=0,
                          activebackground="#e1e8df", bg="#000000", relief="raised")
    scrollbar.pack(side="left", fill="y")

    scroll_frame = tk.Frame(canvas, bg="#e1e8df", pady=20)
    scroll_win = canvas.create_window((0, 0), window=scroll_frame, anchor='nw')

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda event, canvas=canvas, win=scroll_win: on_configure(event, canvas, win))


    tk.Label(scroll_frame, text="* Required Fields", bg="#e1e8df", fg="black", font="Arial 13 bold").pack()
    # Adding Input Fields
    input_fields = ("Name", "Father's Name", "Gender", "DOB(yyyy-mm-dd)", "Crimes Done", "Profile Image")
    ip_len = len(input_fields)
    required = [1, 1, 1, 1, 1, 1]

    entries = []
    for i, field in enumerate(input_fields):
        print()
        row = tk.Frame(scroll_frame, bg="#e1e8df")
        row.pack(side="top", fill="x", pady=15)

        label = tk.Text(row, width=20, height=1, bg="#e1e8df", fg="black", font="Arial 13", highlightthickness=0, bd=0)
        label.insert("insert", field)
        label.pack(side="left")

        if(required[i] == 1):
            label.tag_configure("star", foreground="yellow", font="Arial 13 bold")
            label.insert("end", "  *", "star")
        label.configure(state="disabled")

        if(i != ip_len-1):
            ent = tk.Entry(row, font="Arial 13", selectbackground="#90ceff")
            ent.pack(side="right", expand="true", fill="x", padx=10)
            entries.append((field, ent))
        else:
            menu_var.set("Image 1")
            choices = ["Image 1"]
            opt_menu = tk.OptionMenu(row, menu_var, *choices)
            opt_menu.pack(side="right", fill="x", expand="true", padx=10)
            opt_menu.configure(font="Arial 13", bg="#000000", fg="white", bd=0, highlightthickness=0, activebackground="#e1e8df")
            menu = opt_menu.nametowidget(opt_menu.menuname)
            menu.configure(font="Arial 13", bg="white", activebackground="#90ceff", bd=0)

    # print(entries)

    tk.Button(scroll_frame, text="Register", command=lambda: register(entries, required, menu_var), font="Arial 15 bold",
           bg="#000000", fg="white", pady=10, padx=30, bd=0, highlightthickness=0, activebackground="#e1e8df",
           activeforeground="white").pack(pady=25)


from PIL import Image, ImageTk

def showCriminalProfile(name):
    top = tk.Toplevel(bg="#202d42")
    top.title("Profile")
    top.geometry("1500x900+%d+%d" % (root.winfo_x() + 10, root.winfo_y() + 10))

    tk.Label(top, text=" Profile", fg="white", bg="#202d42", font="Arial 20 bold", pady=10).pack()

    content = tk.Frame(top, bg="#202d42", pady=20)
    content.pack(expand="true", fill="both")
    content.grid_columnconfigure(0, weight=1)  # Adjust the weight to make the left frame expandable
    content.grid_columnconfigure(1, weight=3)  # Adjust the weight to make the right frame expandable
    content.grid_rowconfigure(0, weight=1)

    (id, crim_data) = retrieveData(name)

    # Create the left frame for the profile picture
    left_frame = tk.Frame(content, bg="#202d42")
    left_frame.grid(row=0, column=0, sticky='nsew', padx=15)
    left_frame.grid_propagate(False)  # Prevent the frame from shrinking to fit its contents

    # Load and display the profile picture with increased size
    profile_pic_path = os.path.join("profile_pics", f"{name}.png")  # Assuming the profile pic is saved as name.png
    if os.path.exists(profile_pic_path):
        profile_img = Image.open(profile_pic_path)
        profile_img = profile_img.resize((400, 400))  # Resize the image without using ANTIALIAS
        profile_img = ImageTk.PhotoImage(profile_img)
        profile_pic_label = tk.Label(left_frame, image=profile_img, bg="#202d42")
        profile_pic_label.image = profile_img  # Keep a reference to prevent garbage collection
        profile_pic_label.pack(expand=True, pady=20)
    else:
        tk.Label(left_frame, text="Profile picture not found", fg="red", bg="#202d42", font="Arial 15").pack(pady=20)

    # Create the right frame for displaying details
    right_frame = tk.Frame(content, bg="#202d42")
    right_frame.grid(row=0, column=1, sticky='nsew', padx=(200, 20), pady=(200, 50))  # Increased top and bottom padding and adjusted left and right padding
    right_frame.grid_propagate(False)  # Prevent the frame from shrinking to fit its contents


    for i, item in enumerate(crim_data.items()):
        tk.Label(right_frame, text=item[0], pady=15, fg="yellow", font="Arial 15 bold", bg="#202d42").grid(row=i, column=0, sticky='w')
        tk.Label(right_frame, text=":", fg="yellow", padx=50, font="Arial 15 bold", bg="#202d42").grid(row=i, column=1)
        val = "---" if (item[1] == "") else item[1]
        tk.Label(right_frame, text=val.capitalize(), fg="white", font="Arial 15", bg="#202d42").grid(row=i, column=2, sticky='w')

# def showCriminalProfile(name):
#     top = tk.Toplevel(bg="#202d42")
#     top.title("Criminal Profile")
#     top.geometry("1500x900+%d+%d"%(root.winfo_x()+10, root.winfo_y()+10))

#     tk.Label(top, text="Criminal Profile", fg="white", bg="#202d42", font="Arial 20 bold", pady=10).pack()

#     content = tk.Frame(top, bg="#202d42", pady=20)
#     content.pack(expand="true", fill="both")
#     content.grid_columnconfigure(0, weight=3, uniform="group1")
#     content.grid_columnconfigure(1, weight=5, uniform="group1")
#     content.grid_rowconfigure(0, weight=1)

#     (id, crim_data) = retrieveData(name)



#     info_frame = tk.Frame(content, bg="#202d42")
#     info_frame.grid(row=0, column=1, sticky='w')

#     for i, item in enumerate(crim_data.items()):
#         tk.Label(info_frame, text=item[0], pady=15, fg="yellow", font="Arial 15 bold", bg="#202d42").grid(row=i, column=0, sticky='w')
#         tk.Label(info_frame, text=":", fg="yellow", padx=50, font="Arial 15 bold", bg="#202d42").grid(row=i, column=1)
#         val = "---" if (item[1]=="") else item[1]
#         tk.Label(info_frame, text=val.capitalize(), fg="white", font="Arial 15", bg="#202d42").grid(row=i, column=2, sticky='w')

#     # Load and display the profile picture
#     profile_pic_path = os.path.join("profile_pics", f"{name}.png")  # Assuming the profile pic is saved as name.png
#     if os.path.exists(profile_pic_path):
#         profile_img = Image.open(profile_pic_path)
#         profile_img.thumbnail((200, 200))
#         profile_img = ImageTk.PhotoImage(profile_img)
#         profile_pic_label = tk.Label(top, image=profile_img)
#         profile_pic_label.image = profile_img  # Keep a reference to prevent garbage collection
#         profile_pic_label.pack(pady=20)
#     else:
#         tk.Label(top, text="Profile picture not found", fg="red", bg="#202d42", font="Arial 15").pack(pady=20)



# def startRecognition():
#     global img_read, img_label

#     if(img_label == None):
#         messagebox.showerror("Error", "No image selected. ")
#         return

#     crims_found_labels = []
#     for wid in right_frame.winfo_children():
#         wid.destroy()

#     frame = cv2.flip(img_read, 1, 0)
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     face_coords = detect_faces(gray_frame)

#     if (len(face_coords) == 0):
#         messagebox.showerror("Error", "Image doesn't contain any face or face is too small.")
#     else:
#         (model, names) = train_model()
#         print('Training Successful. Detecting Faces')
#         (frame, recognized) = recognize_face(model, frame, gray_frame, face_coords, names)

#         img_size = left_frame.winfo_height() - 40
#         frame = cv2.flip(frame, 1, 0)
#         showImage(frame, img_size)

#         if (len(recognized) == 0):
#             messagebox.showerror("Error", "No criminal recognized.")
#             return

#         for i, crim in enumerate(recognized):
#             crims_found_labels.append(tk.Label(right_frame, text=crim[0], bg="orange",
#                                             font="Arial 15 bold", pady=20))
#             crims_found_labels[i].pack(fill="x", padx=20, pady=10)
#             crims_found_labels[i].bind("<Button-1>", lambda e, name=crim[0]:showCriminalProfile(name))


#############################################################

def startRecognition():
    global img_read, img_label

    if(img_label == None):
        messagebox.showerror("Error", "No image selected. ")
        return

    crims_found_labels = []
    for wid in right_frame.winfo_children():
        wid.destroy()

    frame = cv2.flip(img_read, 1, 0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, 1, 0)
    face_coords = detect_faces2(gray_frame)

    if (len(face_coords) == 0):
        messagebox.showerror("Error", "Image doesn't contain any face or face is too small.")
    else:
        (model, names) = train_model2()
        print('Training Successful. Detecting Faces')
        (frame, recognized) = recognize_face2(model, frame, gray_frame, face_coords, names)

        img_size = left_frame.winfo_height() - 40
        showImage(frame, img_size)

        if (len(recognized) == 0):
            
            # Display orange rectangle box
            orange_rect = (0, 0, frame.shape[1], frame.shape[0])  # Entire frame
            cv2.rectangle(frame, (orange_rect[0], orange_rect[1]), (orange_rect[2], orange_rect[3]), (0, 165, 255), 2)  # Orange color in BGR
            img_size = left_frame.winfo_height() - 40
            showImage(frame, img_size)
            # Display message box with register button
            register_btn = tk.Button(right_frame, text="Register", command=getPage1, bg="orange", font="Arial 15 bold underline", pady=20)
            register_btn.pack(fill="x", padx=20, pady=10)

            return

        # for i, crim in enumerate(recognized):
        #     crims_found_labels.append(tk.Label(right_frame, text=crim[0], bg="orange",
        #                                     font="Arial 15 bold", pady=20))
        #     crims_found_labels[i].pack(fill="x", padx=20, pady=10)
        #     crims_found_labels[i].bind("<Button-1>", lambda e, name=crim[0]:showCriminalProfile(name))

        for i, crim in enumerate(recognized):
            _, crim_data = retrieveData(crim[0].lower())
            if "Crimes" in crim_data:
                crimes = int(crim_data["Crimes"])
                color = "green" if crimes == 0 else "red"
                crims_found_labels.append(tk.Label(right_frame, text=crim[0], bg=color, font="Arial 15 bold underline", pady=20))
                crims_found_labels[i].pack(fill="x", padx=20, pady=10)
                crims_found_labels[i].bind("<Button-1>", lambda e, name=crim[0]:showCriminalProfile(name))
            else:
                print(f"Crimes data not found for {crim[0]}")




#############################################################



def selectImage():
    global left_frame, img_label, img_read
    for wid in right_frame.winfo_children():
        wid.destroy()

    filetype = [("images", "*.jpg *.jpeg *.png")]
    path = filedialog.askopenfilename(title="Choose a image", filetypes=filetype)

    if(len(path) > 0):
        img_read = cv2.imread(path)

        img_size =  left_frame.winfo_height() - 40
        showImage(img_read, img_size)


## Detection Page ##
def getPage2():
    global active_page, left_frame, right_frame, img_label, heading
    img_label = None
    active_page = 2
    pages[2].lift()

    basicPageSetup(2)
    heading.configure(text="Image Surveillance")
    right_frame.configure(text="Detected Persons Details", fg="black")

    btn_grid = tk.Frame(left_frame, bg="#e1e8df")
    btn_grid.pack()

    tk.Button(btn_grid, text="Select Image", command=selectImage, font="Arial 15 bold", padx=20, bg="#000000",
            fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#e1e8df",
            activeforeground="white").grid(row=0, column=0, padx=25, pady=25)
    tk.Button(btn_grid, text="Recognize", command=startRecognition, font="Arial 15 bold", padx=20, bg="#000000",
           fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#e1e8df",
           activeforeground="white").grid(row=0, column=1, padx=25, pady=25)

# def path_leaf(path):
#     head,tail = ntpath.split(path)


def videoLoop(path,model, names):
    p=path
    q=ntpath.basename(p)
    filenam, file_extension = os.path.splitext(q)
    # print(filename)
    global thread_event, left_frame, webcam, img_label
    start=time.time()
    webcam = cv2.VideoCapture(p)
    old_recognized = []
    crims_found_labels = []
    times = []
    img_label = None
    field=['S.No.', 'Name', 'Time']
    g=filenam+'.csv'
    # filename = "g.csv"
    filename = g
    # with open('people.csv', 'w', ) as csvfile:
    # peoplewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # os.path.join(path, vid.split('.')[0]+'_'+str(count)+'.png'
    num=0
    try:
        # with open('people_Details.csv', 'w', ) as csvfile:
        with open(filename, 'w') as csvfile:
            # peoplewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(field)   
            while not thread_event.is_set():
                
                # Loop until the camera is working
                
                    
                    while (True):
                        # Put the image from the webcam into 'frame'
                        (return_val, frame) = webcam.read()
                        if (return_val == True):
                            break
                        # else:
                        #     print("Failed to open webcam. Trying again...")

                    # Flip the image (optional)
                    frame = cv2.flip(frame, 1, 0)
                    # Convert frame to grayscale
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect Faces
                    face_coords = detect_faces(gray_frame)
                    (frame, recognized) = recognize_face(model, frame, gray_frame, face_coords, names)

                    # Recognize Faces
                    recog_names = [item[0] for item in recognized]
                    if(recog_names != old_recognized):
                        for wid in right_frame.winfo_children():
                            wid.destroy()
                        del(crims_found_labels[:])

                        for i, crim in enumerate(recognized):
                            num += 1
                            x = time.time() - start
                            
                            # Retrieve criminal data
                            _, crim_data = retrieveData(crim[0].lower())
                            
                            # Check if Crimes data exists in the retrieved data
                            if "Crimes" in crim_data:
                                crimes = int(crim_data["Crimes"])
                                color = "green" if crimes == 0 else "red"
                            else:
                                print(f"Crimes data not found for {crim[0]}")
                                # Default to orange if crimes data is not found
                                color = "orange"
                            
                            # Create the label with the appropriate text and color
                            crims_found_labels.append(tk.Label(right_frame, text=crim[0], bg=color,
                                                                font="Arial 15 bold", pady=20))
                            
                            # Pack the label into right_frame
                            crims_found_labels[i].pack(fill="x", padx=20, pady=10)
                            
                            # Bind the label to the showCriminalProfile function
                            crims_found_labels[i].bind("<Button-1>", lambda e, name=crim[0]: showCriminalProfile(name))
                            
                            # Save the data to CSV
                            y = crim[0]
                            print(x, y)
                            arr = [num, y, x]
                            # peoplewriter.writerow(arr)
                            csvwriter.writerow(arr)  

                            
                            # print('hello')
                        old_recognized = recog_names

                    # Display Video stream
                    img_size = min(left_frame.winfo_width(), left_frame.winfo_height()) - 20

                    showImage(frame, img_size)

    except RuntimeError:
        print("[INFO]Caught Runtime Error")
    except tk.TclError:
        print("[INFO]Caught Tcl Error")

##############################################
        
def videoLoop2(model, names):
    filenam='jagadeesh'
    global thread_event, left_frame, webcam, img_label
    start=time.time()
    webcam = cv2.VideoCapture(0)
    old_recognized = []
    crims_found_labels = []
    times = []
    img_label = None
    field=['S.No.', 'Name', 'Time']
    g=filenam+'.csv'
    # filename = "g.csv"
    filename = g
    # with open('people.csv', 'w', ) as csvfile:
    # peoplewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # os.path.join(path, vid.split('.')[0]+'_'+str(count)+'.png'
    num=0
    try:
        # with open('people_Details.csv', 'w', ) as csvfile:
        with open(filename, 'w') as csvfile:
            # peoplewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(field)   
            while not thread_event.is_set():
                
                # Loop until the camera is working
                
                    
                    while (True):
                        # Put the image from the webcam into 'frame'
                        (return_val, frame) = webcam.read()
                        if (return_val == True):
                            break
                        # else:
                        #     print("Failed to open webcam. Trying again...")

                    # Flip the image (optional)
                    frame = cv2.flip(frame, 1, 0)
                    # Convert frame to grayscale
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect Faces
                    face_coords = detect_faces(gray_frame)
                    (frame, recognized) = recognize_face(model, frame, gray_frame, face_coords, names)

                    # Recognize Faces
                    recog_names = [item[0] for item in recognized]
                    if(recog_names != old_recognized):
                        for wid in right_frame.winfo_children():
                            wid.destroy()
                        del(crims_found_labels[:])

                        for i, crim in enumerate(recognized):
                            num += 1
                            x = time.time() - start
                            
                            # Retrieve criminal data
                            _, crim_data = retrieveData(crim[0].lower())
                            
                            # Check if Crimes data exists in the retrieved data
                            if "Crimes" in crim_data:
                                crimes = int(crim_data["Crimes"])
                                color = "green" if crimes == 0 else "red"
                            else:
                                print(f"Crimes data not found for {crim[0]}")
                                # Default to orange if crimes data is not found
                                color = "orange"
                            
                            # Create the label with the appropriate text and color
                            crims_found_labels.append(tk.Label(right_frame, text=crim[0], bg=color,
                                                                font="Arial 15 bold", pady=20))
                            
                            # Pack the label into right_frame
                            crims_found_labels[i].pack(fill="x", padx=20, pady=10)
                            
                            # Bind the label to the showCriminalProfile function
                            crims_found_labels[i].bind("<Button-1>", lambda e, name=crim[0]: showCriminalProfile(name))
                            
                            # Save the data to CSV
                            y = crim[0]
                            print(x, y)
                            arr = [num, y, x]
                            # peoplewriter.writerow(arr)
                            csvwriter.writerow(arr)  

                            
                            # print('hello')
                        old_recognized = recog_names

                    # Display Video stream
                    img_size = min(left_frame.winfo_width(), left_frame.winfo_height()) - 20

                    showImage(frame, img_size)

    except RuntimeError:
        print("[INFO]Caught Runtime Error")
    except tk.TclError:
        print("[INFO]Caught Tcl Error")



# video surveillance Page ##
def getPage4(path):
    p=path
    # print(p)
    global active_page, video_loop, left_frame, right_frame, thread_event, heading
    active_page = 4
    pages[4].lift()

    basicPageSetup(4)
    heading.configure(text="Video Surveillance")
    right_frame.configure(text="Detected Criminals")
    left_frame.configure(pady=40)

    btn_grid = tk.Frame(right_frame, bg="black")
    btn_grid.pack()

    (model, names) = train_model()
    print('Training Successful. Detecting Faces')

    thread_event = threading.Event()
    thread = threading.Thread(target=videoLoop, args=(p,model, names))
    thread.start()
# Define a boolean variable to track if the video has been selected
    
########################################################################
    
# video surveillance Page ##
def getPage6():
    # print(p)
    global active_page, video_loop2, left_frame, right_frame, thread_event, heading
    active_page = 6
    pages[6].lift()

    basicPageSetup(6)
    heading.configure(text="Web Surveillance")
    right_frame.configure(text="Detected Person details")
    left_frame.configure(pady=40)

    btn_grid = tk.Frame(right_frame, bg="#e1e8df")
    btn_grid.pack()

    (model, names) = train_model()
    print('Training Successful. Detecting Faces')

    thread_event = threading.Event()
    thread = threading.Thread(target=videoLoop2, args=(model, names))
    thread.start()

# Define a boolean variable to track if the video has been selected
    
c=0
def getPage3():
    global active_page, video_loop, left_frame, right_frame, thread_event, heading ,c
    
    active_page = 3
    
    pages[3].lift()
    if c==0:
        basicPageSetup(3)
        c=c+1

    # heading.configure(text="Web Surveillance")
    


        btn_grid = tk.Frame(left_frame,bg="#e1e8df")
        btn_grid.pack()

        tk.Button(btn_grid, text="Select Video", command=selectvideo, font="Arial 15 bold", padx=20, bg="#000000",
                fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#e1e8df",
                activeforeground="white").grid(row=0, column=0, padx=25, pady=25)
    
######################################################
    
b=0
def getPage5():
    global active_page, video_loop, left_frame, right_frame, thread_event, heading ,b
    
    active_page = 5
    
    pages[5].lift()
    if b==0:
        basicPageSetup(5)
        b=b+1

    # heading.configure(text="Web Surveillance")
    


        btn_grid = tk.Frame(left_frame,bg="#e1e8df")
        btn_grid.pack()

        tk.Button(btn_grid, text="Web Surveilliance", command=getPage6, font="Arial 15 bold", padx=20, bg="#000000",
                fg="white", pady=10, bd=0, highlightthickness=0, activebackground="#e1e8df",
                activeforeground="white").grid(row=0, column=0, padx=25, pady=25)
    

##########################################




##########################################
def selectvideo():
    global left_frame, img_label, img_read
    filetype = [("video", "*.mp4 *.mkv")]
    path = filedialog.askopenfilename(title="Choose a video", filetypes=filetype)
    p=''
    p=path
    
    if(len(path) > 0):
        # vid_read = cv2.imread(path)
        # print(vid_read)
        getPage4(p)
        # img_read = cv2.imread(path)
    #     img_size =  left_frame.winfo_height() - 40
    #     showImage(img_read, img_size)
##################################################
        


def selectvideo1():
    # global left_frame, img_label, img_read
    # for wid in right_frame.winfo_children():
    #     wid.destroy()

    filetype = [("video", "*.mp4 *.mkv")]
    path = filedialog.askopenfilename(title="Choose a video", filetypes=filetype)
    p=''
    p=path
    
    if(len(path) > 0):
        # vid_read = cv2.imread(path)
        # print(vid_read)
       detect(p)


# ######################################## Home Page ####################################
tk.Label(pages[0], text="EPIANN", fg="black", bg="#e1e8df", font="RamaRaja 40 bold", pady=60).pack()

logo = tk.PhotoImage(file=r"E:\FinalYearProjectDetails\MajorprojectFacialRecognition\Facial-Recognition-for-Crime-Detection\colours.png")
logo_label = tk.Label(pages[0], image=logo, bd=0)
logo_label.pack(side='bottom', padx=100, pady=50, anchor='sw')  # Adjust padx and pady values as needed



btn_frame = tk.Frame(pages[0], bg="#e1e8df", pady=60)
btn_frame.pack()
btn_frame.pack(pady=(20,40))



# Three buttons on the left
# tk.Button(btn_frame, text="Input Video", command=selectvideo1).grid(row=2, column=0, pady=(10,30), padx=(40, 250))
tk.Button(btn_frame, text="Register a Person", command=getPage1).grid(row=2, column=0, pady=(10,30), padx=(40, 200))
tk.Button(btn_frame, text="Image Surveillance", command=getPage2).grid(row=8, column=0, pady=(30,30), padx=(40, 200))

# Two buttons on the right
tk.Button(btn_frame, text="Video Surveillance", command=getPage3).grid(row=2, column=1, pady=(10,30), padx=(40, 200))
tk.Button(btn_frame, text="Web Surveillance", command=getPage5).grid(row=8, column=1, pady=(30,30), padx=(40, 200))

# Styling buttons
for btn in btn_frame.winfo_children():
    btn.configure(font="Arial 25", width=20, bg="white", fg="black",
                  bd=0, highlightthickness=0, activebackground="white", activeforeground="black")


pages[0].lift()
root.mainloop()