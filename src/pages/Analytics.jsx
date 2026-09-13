import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis, LineChart, Line, Legend
} from 'recharts'
import PageHeader from '../components/PageHeader'
import ChartCard from '../components/ChartCard'
import { categoryData, priorityData, responseData, confidenceData, activityData } from '../data/mockData'

const COLORS = ['#6C4DFF','#8D76FF','#A994FF','#5FE7F2','#4E35D3','#C9C5FF']
const PRIORITY = ['#A994FF','#6C4DFF','#FF7A98','#FF4F78']

const tipStyle = { background:'rgba(255,255,255,.82)', border:'1px solid rgba(255,255,255,.9)', borderRadius:14, boxShadow:'0 16px 40px rgba(75,53,160,.14)', backdropFilter:'blur(18px)' }

export default function Analytics() {
  return (
    <>
      <PageHeader title="Analytics" text="Performance, confidence and support-flow intelligence across your AI operation."/>
      <div className="analytics-grid">
        <ChartCard title="Ticket categories" subtitle="Share of classified tickets">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart><Pie data={categoryData} dataKey="value" nameKey="name" innerRadius={65} outerRadius={95} paddingAngle={4}>
              {categoryData.map((_,i)=><Cell key={i} fill={COLORS[i%COLORS.length]}/>)}
            </Pie><Tooltip contentStyle={tipStyle}/><Legend/></PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Priority distribution" subtitle="Current queue composition">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={priorityData}><CartesianGrid stroke="rgba(90,72,180,.09)" vertical={false}/><XAxis dataKey="name"/><YAxis/>
              <Tooltip contentStyle={tipStyle}/><Bar dataKey="value" radius={[10,10,0,0]}>
                {priorityData.map((_,i)=><Cell key={i} fill={PRIORITY[i]}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Median response time" subtitle="Minutes by day" className="wide-chart">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={responseData}><defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#6C4DFF" stopOpacity=".45"/><stop offset="100%" stopColor="#6C4DFF" stopOpacity=".03"/></linearGradient></defs>
              <CartesianGrid stroke="rgba(90,72,180,.08)" vertical={false}/><XAxis dataKey="day"/><YAxis/><Tooltip contentStyle={tipStyle}/>
              <Area type="monotone" dataKey="minutes" stroke="#6C4DFF" strokeWidth={3} fill="url(#area)"/>
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Confidence scores" subtitle="Tickets by confidence band">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={confidenceData}><CartesianGrid stroke="rgba(90,72,180,.08)" vertical={false}/><XAxis dataKey="range"/><YAxis/><Tooltip contentStyle={tipStyle}/>
              <Bar dataKey="tickets" fill="#5FE7F2" radius={[10,10,0,0]}/>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Recent classification activity" subtitle="Classified vs manually reviewed" className="wide-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={activityData}><CartesianGrid stroke="rgba(90,72,180,.08)" vertical={false}/><XAxis dataKey="time"/><YAxis/><Tooltip contentStyle={tipStyle}/><Legend/>
              <Line type="monotone" dataKey="classified" stroke="#6C4DFF" strokeWidth={3} dot={false}/>
              <Line type="monotone" dataKey="reviewed" stroke="#5FE7F2" strokeWidth={3} dot={false}/>
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </>
  )
}
