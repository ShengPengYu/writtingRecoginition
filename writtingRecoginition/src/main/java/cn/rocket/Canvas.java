package cn.rocket;
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.ArrayList;

public class Canvas extends JPanel implements MouseListener,MouseMotionListener{
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int x = 0;
    private int y = 0;

    //record line list
    private ArrayList<Integer[]> lineLlist = new ArrayList<Integer[]>();

    //outline rectangle record
    private int outlineX = -1;
    private int outlineY = -1;
    private int outlineWidth = -1;
    private int outlineHeight = -1;
    private int outset = -1;

    public void resetOutLine(){
        this.outlineX = -1;
        this.outlineY = -1;
        this.outlineWidth = -1;
        this.outlineHeight = -1;
        this.outset = -1;
    }

    public void setOutLine(final int x,final int y,final int width,final int height){
        new Thread(new Runnable() {
            public void run() {
                for(int i=50;i>=0;i--){
                    try {
                        outlineX = x-i;
                        outlineY = y-i;
                        outlineWidth = width+i*2;
                        outlineHeight = height+i*2;
                        outset = 1;
                        repaint();
                        Thread.sleep(5);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

            }
        }).start();
    }

    public Canvas(int width, int height){
        this.setBackground(Color.BLACK);
        this.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        this.setVisible(true);

        this.addMouseListener(this);
        this.addMouseMotionListener(this);
    }

    public void clear(){
        this.lineLlist.clear();
        this.resetOutLine();
        this.repaint();
    }

    public void paint(Graphics graphics){
        super.paint(graphics);
        Graphics2D g2d = (Graphics2D)graphics;

        //background is white
        g2d.setBackground(Color.WHITE);

        //draw line
        g2d.setColor(Color.WHITE);
        g2d.setStroke(new BasicStroke(20));
        for(int i=0;i<lineLlist.size();i++){
            g2d.drawLine(lineLlist.get(i)[0],lineLlist.get(i)[1],
                    lineLlist.get(i)[2],lineLlist.get(i)[3]);
        }

        if(outset != -1){
            g2d.setColor(Color.red);
            g2d.setStroke(new BasicStroke(2));
            g2d.drawRect(outlineX,outlineY,outlineWidth,outlineHeight);
        }

        g2d.dispose();
    }

    public void mouseClicked(MouseEvent e) {
    }

    public void mousePressed(MouseEvent e) {
        this.x = e.getX();
        this.y = e.getY();
    }

    public void mouseReleased(MouseEvent e) {
    }

    public void mouseEntered(MouseEvent e) {
    }

    public void mouseExited(MouseEvent e) {
    }

    public void mouseDragged(MouseEvent e) {
        lineLlist.add(new Integer[]{this.x,this.y,e.getX(),e.getY()});
        this.x = e.getX();
        this.y = e.getY();

        this.repaint();
    }

    public void mouseMoved(MouseEvent e) {
    }
}
