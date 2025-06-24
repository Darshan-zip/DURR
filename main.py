import pyodbc
import xml.etree.ElementTree as ET
import pandas as pd
from flask import Flask,render_template,request,jsonify,make_response
import pandas as pd
from datetime import datetime, timedelta

def decode_yydddHH(val):
    try:
        val = str(val).zfill(7)  # Make sure it's 7 digits, pad with leading 0s if needed
        yy = int(val[:2])
        ddd = int(val[2:5])
        hh = int(val[5:7])

        # Convert YY to full year (assume 2000s)
        year = 2000 + yy if yy < 50 else 1900 + yy

        # Get the datetime for Jan 1st of that year + ddd days - 1
        dt = datetime(year, 1, 1) + timedelta(days=ddd - 1, hours=hh)
        return dt
    except Exception as e:
        print(f"Error decoding {val}: {e}")
        return pd.NaT

app=Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')
@app.route("/inc",methods=['POST'])
def inc():
    req=request.get_json()
    server=req['server']
    database=req['database']
    username=req['username']
    password=req['password']
    xml_path=req['XML']
    file_path = xml_path.replace("\\", "/")
    fromDate=req['fromDate']
    toDate=req['toDate']
  
    
    conn_str = f"""
    DRIVER={{SQL Server}};
    SERVER={server};
    DATABASE={database};
    Trusted_Connection=yes;
    """

    conn = pyodbc.connect(
        conn_str
    )

    


    print(file_path)
    
    # Parse the XML file
    tree = ET.parse(file_path)  
    root = tree.getroot()
    data = []
    for entry in root.find('entries').findall('entry'):
        entry_id = entry.attrib.get('id')  
        
        config = entry.find('config')
        if config is not None:
            unit = config.attrib.get('unit')
            areakey = config.attrib.get('areakey')
            medium = config.attrib.get('medium')
            
            data.append({
                'id': entry_id,
                'unit': unit,
                'areakey': areakey,
                'medium': medium
            })


    df = pd.DataFrame(data)
    query="select value_key,value_id as id,value_summary from New_values"
    df_sql = pd.read_sql(query, conn)
    
    df_sql['id'] = df_sql['id'].astype(str)
    df['id'] = df['id'].astype(str)
    df_sql_date = df_sql[(df_sql['value_key'] >= int(fromDate)) & (df_sql['value_key'] <= int(toDate))]

    
    df_sql_date['date & time'] = df_sql_date['value_key'].apply(decode_yydddHH).dt.strftime('%Y-%m-%d %H:%M')

    result_df = df_sql_date.merge(df, how='left', on='id')
    val_sum=result_df['value_summary'].sum()
    val_sum=round(val_sum, 2)
    html_table = result_df.to_html(index=False, classes="display", table_id="myTable")
   
    res=make_response(jsonify({"table":html_table,"valSum":val_sum}),200)
    return res

    
if __name__=='__main__':
        app.run(debug=True)
