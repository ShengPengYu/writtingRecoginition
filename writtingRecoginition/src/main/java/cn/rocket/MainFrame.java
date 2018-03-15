
package cn.rocket;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.apache.commons.lang3.ArrayUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class MainFrame extends JFrame {
	private static final long serialVersionUID = 1L;
	
	private Canvas canvas ;
	
	private   MultiLayerNetwork model;
	ArrayList<?> list = new ArrayList<Object>();

	public static void main(String[] args) throws IOException {
		MainFrame drawBorder = new MainFrame();
		drawBorder.initFrame();
	}

	public void initFrame() throws IOException {
		
		 model = ModelSerializer.restoreMultiLayerNetwork("trained_mnist_model.zip");
		 
		
		this.setTitle("handwriting numerals recognition");
		this.setDefaultCloseOperation(3);
		this.setLocationRelativeTo(null);
		this.setResizable(false);
		
		JPanel panel = new JPanel();
		panel.setLayout(new BorderLayout());
		this.add(panel);

		
		canvas  = new Canvas(680,680);
		canvas.setPreferredSize(new Dimension(280,280));
		 this.canvas.setBounds(new Rectangle(85, 30, 280, 280));
		panel.add(canvas,BorderLayout.CENTER);
		
		
		JPanel predictPanel = new JPanel();
		predictPanel.setLayout(new BorderLayout());
		predictPanel.setBackground(Color.CYAN);
		predictPanel.setPreferredSize(new Dimension(200, 280));
		panel.add(predictPanel,BorderLayout.EAST);
		
		
		JLabel tip = new JLabel("Identify results:");
		tip.setFont(new Font("Times New Roman", Font.BOLD, 20));
		//tip.setForeground(Color.WHITE);
		predictPanel.add(tip,BorderLayout.NORTH);
		
		JLabel show = new JLabel("");
		show.setFont(new Font("Times New Roman", Font.BOLD, 100));
		//show.setForeground(Color.WHITE);
		predictPanel.add(show,BorderLayout.CENTER);
		
		
		JLabel lable = new JLabel("CF:");
		lable.setFont(new Font("Times New Roman", Font.BOLD, 20));
		//lable.setForeground(Color.WHITE);
		predictPanel.add(lable,BorderLayout.SOUTH);




		// 主面板添加下方面板
		JPanel paneldown = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
		//paneldown.setPreferredSize(new Dimension(0, 60));
		//paneldown.setLayout(null);
		paneldown.setBackground(Color.gray);
		panel.add(paneldown, BorderLayout.SOUTH);
		
		
		JLabel space = new JLabel("");
		space.setPreferredSize(new Dimension(50, 40));
		paneldown.add(space);
		
		JButton start = new JButton("Reginition");
		start.setFont(new Font("Times New Roman", Font.BOLD, 20));
		start.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
			    try {
			    	
			    	int[] outline = getOutline();
			    	System.out.print(ArrayUtils.toString(outline));
					/*BufferedImage b = new BufferedImage(canvas.getWidth(),canvas.getHeight(),BufferedImage.TYPE_INT_BGR);
					BufferedImage out = new BufferedImage(28, 28, BufferedImage.TYPE_INT_BGR);
					Graphics graphics = b.createGraphics(); 
					canvas.paint(graphics);        
		
					out.getGraphics().drawImage(b.getScaledInstance(28, 28, Image.SCALE_SMOOTH), 0, 0, null);
					OutputStream outputStream = new FileOutputStream(new File("D:\\test.jpeg"));
					ImageIO.write(out, "JPEG", outputStream);
					graphics.dispose();
					outputStream.close();
					System.out.println("Image save finished!");
					// FileChose is a string we will need a file

			        // Use NativeImageLoader to convert to numerical matrix
*/
			        NativeImageLoader loader = new NativeImageLoader(28, 28, 1);

			        // Get the image into an INDarray

			        INDArray image = loader.asMatrix(saveJPanel(outline));

			        // 0-255
			        // 0-1
			        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
			        scaler.transform(image);
			        // Pass through to neural Net
			        
			        INDArray output = model.output(image);

			      
			        //log.info("## List of Labels in Order## ");
			        // In new versions labels are always in order
			        int index = 0;
			        double max = 0.000001;
			        for(int i = 0 ; i < output.columns() ;i++) {
			        	if(output.getDouble(0, i) >= max) {
			        		index = i;
			        		max = output.getDouble(0, i) ;
			        	}
			        }
			        
			        show.setText(index+"");
			        lable.setText("CF:"+max*100 +"%");
			        
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		});
		paneldown.add(start);
	
		JLabel space1 = new JLabel("");
		space1.setPreferredSize(new Dimension(50, 40));
		paneldown.add(space1);
		
		JButton clear = new JButton("Clear Canvas");
		clear.setFont(new Font("Times New Roman", Font.BOLD, 20));
		clear.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				list.clear();
				canvas.clear();
			}
		});
		paneldown.add(clear);

		this.setVisible(true);
		this.pack();
	}
	
	
	 public int[] getOutline(){
	        double[] grayMatrix = ImageUtil.getInstance().getGrayMatrixFromPanel(canvas, null);
	        int[] binaryArray = ImageUtil.getInstance().transGrayToBinaryValue(grayMatrix);
	        int minRow = Integer.MAX_VALUE;
	        int maxRow = Integer.MIN_VALUE;
	        int minCol = Integer.MAX_VALUE;
	        int maxCol = Integer.MIN_VALUE;
	        for(int i=0;i<binaryArray.length;i++){
	            int row = i/28;
	            int col = i%28;
	            if(binaryArray[i] == 0){
	                if(minRow > row){
	                    minRow = row;
	                }
	                if(maxRow < row){
	                    maxRow = row;
	                }
	                if(minCol > col){
	                    minCol = col;
	                }
	                if(maxCol < col){
	                    maxCol = col;
	                }
	            }
	        }
	        int len = Math.max((maxCol-minCol+1)*10, (maxRow-minRow+1)*10);

	        int p = 0 ;
	        p = (len+40 - (maxCol-minCol+1)*10-20-20)/2;
	        if(p<0) p = 0;
	        
	       int x = minCol*10-20-p ;
	       int y = minRow*10-20   ;
	       int width = len+40 ;
	       if(x<0 || y <0){
	    	    x = minCol*10 ;
		        y = minRow*10 ;
		        width = len ;
	       } 
	       canvas.setOutLine(x, y, width,width );
	        
	        return new int[]{x, y, width,width};
	    }

	 
	  public BufferedImage saveJPanel(int[] outline){
	        Dimension imageSize = this.canvas.getSize();
	        BufferedImage image = new BufferedImage(imageSize.width,imageSize.height, BufferedImage.TYPE_INT_RGB);
	        Graphics2D graphics = image.createGraphics();
	        this.canvas.paint(graphics);
	        graphics.dispose();
	        try {
	            //cut
	            if(outline[0] + outline[2] > canvas.getWidth()){
	                outline[2] = canvas.getWidth()-outline[0];
	            }
	            if(outline[1] + outline[3] > canvas.getHeight()){
	                outline[3] = canvas.getHeight()-outline[1];
	            }
	            image = image.getSubimage(outline[0],outline[1],outline[2],outline[3]);
	            //resize
	            Image smallImage = image.getScaledInstance(Constant.smallWidth, Constant.smallHeight, Image.SCALE_SMOOTH);
	            BufferedImage bSmallImage = new BufferedImage(Constant.smallWidth,Constant.smallHeight,BufferedImage.TYPE_INT_RGB);
	            Graphics graphics1 = bSmallImage.getGraphics();
	            graphics1.drawImage(smallImage, 0, 0, null);
	            graphics1.dispose();

	            return bSmallImage;
	        } catch (Exception e) {
	            e.printStackTrace();
	        }
	        return null;
	    }
	   
	    
}
