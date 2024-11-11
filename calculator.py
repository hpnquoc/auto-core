import sys
import math
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLineEdit, QPushButton, QWidget, QGridLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class ScientificCalculatorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Scientific Calculator")
        self.setFixedSize(500, 600)

        # Initialize the main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Display for the calculator
        self.display = QLineEdit()
        self.display.setReadOnly(True)
        self.display.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.display.setFont(QFont("Arial", 20))
        self.layout.addWidget(self.display)

        # Create button grid layout
        self.button_grid = QGridLayout()
        self.layout.addLayout(self.button_grid)

        # Define buttons and add them to the grid
        buttons = [
            ('7', 0, 0), ('8', 0, 1), ('9', 0, 2), ('/', 0, 3), ('√', 0, 4), ('(', 0, 5), (')', 0, 6),
            ('4', 1, 0), ('5', 1, 1), ('6', 1, 2), ('*', 1, 3), ('x²', 1, 4), ('%', 1, 5), ('C', 1, 6),
            ('1', 2, 0), ('2', 2, 1), ('3', 2, 2), ('-', 2, 3), ('1/x', 2, 4), ('log', 2, 5), ('ln', 2, 6),
            ('0', 3, 0), ('.', 3, 1), ('=', 3, 2), ('+', 3, 3), ('e^x', 3, 4), ('π', 3, 5), ('e', 3, 6),
            ('sin', 4, 0), ('cos', 4, 1), ('tan', 4, 2), ('n!', 4, 3), ('x^y', 4, 4), ('Q', 4, 5)
        ]

        for text, row, col in buttons:
            button = QPushButton(text)
            button.setFont(QFont("Arial", 14))
            button.setFixedSize(60, 60)
            button.clicked.connect(lambda _, t=text: self.on_button_click(t))
            self.button_grid.addWidget(button, row, col)

        # Set up the expression variable
        self.expression = ""

    def on_button_click(self, button_text):
        if button_text == 'C':
            self.expression = ""
            self.display.setText("")
        elif button_text == '=':
            self.evaluate_expression()
        elif button_text == '√':
            self.calculate_sqrt()
        elif button_text == 'x²':
            self.calculate_square()
        elif button_text == '1/x':
            self.calculate_reciprocal()
        elif button_text == '%':
            self.calculate_percentage()
        elif button_text == 'ln':
            self.calculate_ln()
        elif button_text == 'log':
            self.calculate_log()
        elif button_text == 'e^x':
            self.calculate_exp()
        elif button_text == 'n!':
            self.calculate_factorial()
        elif button_text == 'π':
            self.insert_constant(math.pi)
        elif button_text == 'e':
            self.insert_constant(math.e)
        elif button_text in ('sin', 'cos', 'tan'):
            self.calculate_trig_function(button_text)
        elif button_text == 'x^y':
            self.expression += '**'
            self.display.setText(self.expression)
        elif button_text == 'Q':
            self.close()
        else:
            self.expression += button_text
            self.display.setText(self.expression)

    def evaluate_expression(self):
        try:
            result = str(eval(self.expression))
            self.display.setText(result)
            self.expression = result
        except:
            self.display.setText("Error")
            self.expression = ""

    def calculate_sqrt(self):
        try:
            result = str(math.sqrt(float(self.expression)))
            self.display.setText(result)
            self.expression = result
        except:
            self.display.setText("Error")
            self.expression = ""

    def calculate_square(self):
        try:
            result = str(float(self.expression) ** 2)
            self.display.setText(result)
            self.expression = result
        except:
            self.display.setText("Error")
            self.expression = ""

    def calculate_reciprocal(self):
        try:
            result = str(1 / float(self.expression))
            self.display.setText(result)
            self.expression = result
        except:
            self.display.setText("Error")
            self.expression = ""

    def calculate_percentage(self):
        try:
            result = str(float(self.expression) / 100)
            self.display.setText(result)
            self.expression = result
        except:
            self.display.setText("Error")
            self.expression = ""

    def calculate_ln(self):
        try:
            result = str(math.log(float(self.expression)))
            self.display.setText(result)
            self.expression = result
        except:
            self.display.setText("Error")
            self.expression = ""

    def calculate_log(self):
        try:
            result = str(math.log10(float(self.expression)))
            self.display.setText(result)
            self.expression = result
        except:
            self.display.setText("Error")
            self.expression = ""

    def calculate_exp(self):
        try:
            result = str(math.exp(float(self.expression)))
            self.display.setText(result)
            self.expression = result
        except:
            self.display.setText("Error")
            self.expression = ""

    def calculate_factorial(self):
        try:
            result = str(math.factorial(int(float(self.expression))))
            self.display.setText(result)
            self.expression = result
        except:
            self.display.setText("Error")
            self.expression = ""

    def calculate_trig_function(self, func):
        try:
            angle_in_radians = math.radians(float(self.expression))
            if func == 'sin':
                result = str(math.sin(angle_in_radians))
            elif func == 'cos':
                result = str(math.cos(angle_in_radians))
            elif func == 'tan':
                result = str(math.tan(angle_in_radians))
            self.display.setText(result)
            self.expression = result
        except:
            self.display.setText("Error")
            self.expression = ""

    def insert_constant(self, constant_value):
        self.expression = str(constant_value)
        self.display.setText(self.expression)

    def keyPressEvent(self, event):
        key = event.key()
        key_map = {
            Qt.Key.Key_0: '0', Qt.Key.Key_1: '1', Qt.Key.Key_2: '2', Qt.Key.Key_3: '3',
            Qt.Key.Key_4: '4', Qt.Key.Key_5: '5', Qt.Key.Key_6: '6', Qt.Key.Key_7: '7',
            Qt.Key.Key_8: '8', Qt.Key.Key_9: '9', Qt.Key.Key_Plus: '+', Qt.Key.Key_Minus: '-',
            Qt.Key.Key_Asterisk: '*', Qt.Key.Key_Slash: '/', Qt.Key.Key_Equal: '=',
            Qt.Key.Key_Return: '=', Qt.Key.Key_Enter: '=', Qt.Key.Key_Period: '.',
            Qt.Key.Key_C: 'C', Qt.Key.Key_Q: 'Q'
        }
        if key in key_map:
            if key_map[key] == '=':
                self.evaluate_expression()
            elif key_map[key] == 'C':
                self.expression = ""
                self.display.setText("")
            elif key_map[key] == 'Q':
                self.close()
            else:
                self.on_button_click(key_map[key])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    calculator = ScientificCalculatorApp()
    calculator.show()
    sys.exit(app.exec())
