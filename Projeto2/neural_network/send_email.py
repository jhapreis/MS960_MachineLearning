import smtplib
import boto3
from botocore.exceptions import ClientError

try:
    from Projeto2._secure import cfg_email
except:
    print('\nNo config file for the email.\n'+
    'Please, make sure you have a config-file named "cfg_email.py" and with the following files:\n'+
    'EMAIL_ADDRESS, EMAIL_DESTINATION and PASSWORD --- all in the Config folder.\n\n')
    raise

'''
    You need to give permission on your email:
In order for this script to work, you must also enable "less secure apps" to access your Gmail account.
As a warning, this is not ideal, and Google does indeed warn against enabling this feature.
Note: I think that you necessarily need to login on the computer, so google can trust on the device.
https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa3V5M3YzRUZnMlppc1hGbnRoVXJzSXBzN3d5UXxBQ3Jtc0trUG5EbV9qYUI3NGc3U01XQ3RRMmRZWlNGZjY1Wm1WeEVvVGRnek5KRVFRcXN1dm5OUEg0V2ExQXRtRjhFRHJlUXVQZ2FIanVpMXJDdWdIQ2lHNE5iRThQdGhpWngxbWlnaWdMWXNFNEt5QUVCUXZ6cw&q=https%3A%2F%2Fmyaccount.google.com%2Flesssecureapps
'''



def SendEmail(subject, msg):
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(cfg_email.EMAIL_ADDRESS, cfg_email.PASSWORD)
        message = 'Subject: {}\n\n{}'.format(subject, msg)
        server.sendmail(from_addr=cfg_email.EMAIL_ADDRESS, to_addrs=cfg_email.EMAIL_DESTINATION, msg=message)
        server.quit()
        print('Success: Email sent.\n\n\n')
    except:
        print('Email failed to send.\n\n\n')



def SendEmail_2(subject, msg):

    # Replace sender@example.com with your "From" address.
    # This address must be verified with Amazon SES.
    SENDER = f"Sender Name <{cfg_email.EMAIL_ADDRESS}>"

    # Replace recipient@example.com with a "To" address. If your account 
    # is still in the sandbox, this address must be verified.
    RECIPIENT = f"{cfg_email.EMAIL_DESTINATION}"

    # Specify a configuration set. If you do not want to use a configuration
    # set, comment the following variable, and the 
    # ConfigurationSetName=CONFIGURATION_SET argument below.
    CONFIGURATION_SET = "ConfigSet"

    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-west-2"

    # The subject line for the email.
    SUBJECT = "Amazon SES Test (SDK for Python)"

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = ("Amazon SES Test (Python)\r\n"
                "This email was sent with Amazon SES using the "
                "AWS SDK for Python (Boto)."
                )
                
    # The HTML body of the email.
    BODY_HTML = """<html>
    <head></head>
    <body>
    <h1>Amazon SES Test (SDK for Python)</h1>
    <p>This email was sent with
        <a href='https://aws.amazon.com/ses/'>Amazon SES</a> using the
        <a href='https://aws.amazon.com/sdk-for-python/'>
        AWS SDK for Python (Boto)</a>.</p>
    </body>
    </html>
                """            

    # The character encoding for the email.
    CHARSET = "UTF-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses',region_name=AWS_REGION)

    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
            # If you are not using a configuration set, comment or delete the
            # following line
            ConfigurationSetName=CONFIGURATION_SET,
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])

