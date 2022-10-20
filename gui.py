import tkinter as tk
from secure_db_connection import SecureConnect
import webbrowser


class CKD_GUI:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1500x700")
        self.main_window.title('Project Login')
        self.main_window.configure(bg='#333333')
        self._secure_connection = None # private variable for initiating secure connection
        self._url = "https://shubh2016shiv-ckd-prediction-app-jsw9gd.streamlitapp.com/" # private variable for opening URL once secure connection is created with MongoDB DB

        frame = tk.Frame(bg='#333333')

        welcome_label = tk.Label(frame,
                                 text="Welcome to the Project - Detection of Chronic Kidney Disease using Machine Learning",
                                 relief=tk.FLAT, font=("Segoe UI", 18), bg='#333333', fg="#FFFFFF")

        security_message_label = tk.Label(frame,
                                          text="⚠️The access to MongoDB Database is secured and encrypted. \n You only have read-only access. Kindly, enter the username and password to proceed ahead. If access is granted, then application will open in web browser.",
                                          relief=tk.FLAT, font=("Segoe UI", 15), bg='#333333', fg="#93C572")

        username_label = tk.Label(frame, text="Username : ", bg='#333333', fg="#FFFFFF", font=("Times New Roman", 16))
        self.username_entry = tk.Entry(frame, font=("Arial", 16))
        self.password_entry = tk.Entry(frame, show="*", font=("Arial", 16))
        password_label = tk.Label(frame, text="Password : ", bg='#333333', fg="#FFFFFF", font=("Times New Roman", 16))
        open_app_button = tk.Button(frame, text="Open Application", bg="#8E9590", fg="#FFFFFF",
                                    font=("Segoe UI Semibold", 16), command=self.open_app)

        self.grant_access_response = tk.StringVar()
        self.grant_access_response_label = tk.Label(frame, textvariable=self.grant_access_response,
                                                    font=("Segoe UI", 12), bg='#333333', fg="#93C572")

        self.fail_access_response = tk.StringVar()
        self.fail_access_response_label = tk.Label(frame, textvariable=self.fail_access_response,
                                                   font=("Segoe UI", 12), bg='#333333', fg="#EE4B2B")

        # Placing widgets on the screen
        welcome_label.grid(row=0, column=0, columnspan=2, sticky="news", pady=60)
        security_message_label.grid(row=1, column=0, columnspan=2, sticky="news", pady=60)
        username_label.grid(row=2, column=0)
        self.username_entry.grid(row=2, column=1, pady=20)
        password_label.grid(row=3, column=0)
        self.password_entry.grid(row=3, column=1, pady=20)

        open_app_button.grid(row=4, column=0, columnspan=2, pady=10)
        self.grant_access_response_label.grid(row=5, column=0, columnspan=2, pady=20)
        self.fail_access_response_label.grid(row=5, column=0, columnspan=2, pady=30)

        frame.pack()

        tk.mainloop()

    def open_app(self):
        """
        Function to open application in web browser only after authentication is done
        :return: None
        """
        user_name = self.username_entry.get()
        password = self.password_entry.get()
        if (len(user_name) != 0) and (len(password) != 0):
            self._secure_connection = SecureConnect(username=user_name, password=password)
            if self._secure_connection.is_connection_valid():
                self.fail_access_response.set("")
                self.grant_access_response.set("✅ Connection Successful. Opening application in Web Browser")
                webbrowser.open(self._url)
            else:
                self.grant_access_response.set("")
                self.fail_access_response.set("❌ Connection Failed. Either username or password is wrong. Try Again!")
        else:
            self.grant_access_response.set("")
            self.fail_access_response.set("❌ Either username or password is empty. Enter both to proceed ahead")


gui = CKD_GUI()
