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