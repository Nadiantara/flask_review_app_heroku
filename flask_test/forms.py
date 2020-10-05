from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo
from datetime import date
from wtforms.fields.html5 import DateField
from wtforms.fields.html5 import DateTimeField


class DateForm(FlaskForm):
    start_date = DateField('Start Date',default=date.today)
    end_date = DateField('End Date',default=date.today)

    def validate_on_submit(self):
        result = super(DateForm, self).validate()
        if (self.startdate.data>self.enddate.data):
            return False
        else:
            return result

